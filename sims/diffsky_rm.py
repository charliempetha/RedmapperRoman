import os
import pyarrow.dataset as ds
import numpy as np
import healpy as hp
import hdf5plugin
import h5py
from astropy.table import Table


class DiffSkyRedMapper:
    def __init__(self, survey="lsst_roman", z_max=1.0):

        self.data_dir = "/Users/cmpetha/Projects/ExternalData/Simulations/diffsky/e2e/"
        self.input_fname = self.data_dir + "e2e_catalog_noshear.parquet"
        self.survey = survey
        self.z_max = z_max

    def test_e2e_cat(self):
        dataset = ds.dataset(self.input_fname, format="parquet")
        table = dataset.to_table()
        data = table.to_pandas()
        print(data.head())
        print(data.columns)
        print(f"Number of galaxies: {len(data)}")

    def make_input_hdf5_with_flux_and_errs(
        self,
        make_specz_sample=True,
        spec_sample_frac=0.05,
    ):
        """
        Create the final DiffSky cutout HDF5 file for Redmapper.
        """
        # bands names in parquet file
        lsst_bands = ["LSST_u", "LSST_g", "LSST_r", "LSST_i", "LSST_z", "LSST_y"]
        roman_bands = ["Y", "J", "H"]
        # bands names requried by redMaPPerRoman
        lsst_bands_redmapper = ["u", "g", "r", "i", "z", "y"]
        roman_bands_redmapper = ["F106", "F129", "F158"]
        if self.survey == "roman":
            bands = roman_bands
            bands_inds = roman_bands_redmapper
        elif self.survey == "lsst":
            bands = lsst_bands
            bands_inds = lsst_bands_redmapper
        elif self.survey == "lsst_roman":
            bands = lsst_bands + roman_bands
            bands_inds = lsst_bands_redmapper + roman_bands_redmapper
        else:
            raise ValueError(f"Unknown survey: {self.survey}")
        file_suff = f"{self.survey}_zmax{self.z_max}"
        diffsky_rm_file = f"diffsky_{file_suff}.hdf5"

        # Use GOLD fluxes and errors from the parquet file
        needed_cols = ["z", "objectid", "ra", "dec"]
        needed_cols += [f"flux_gold_{b}" for b in bands]
        needed_cols += [f"flux_err_gold_{b}" for b in bands]

        dataset = ds.dataset(self.input_fname, format="parquet")
        table = dataset.to_table(columns=needed_cols, filter=ds.field("z") < self.z_max)
        data = table.to_pandas()
        z = data["z"].to_numpy()
        gal_ids = data["objectid"].to_numpy()
        ra = data["ra"].to_numpy()
        dec = data["dec"].to_numpy()

        if make_specz_sample:
            # Randomly select x% of total sample to be specz
            # Uniform over redshift range
            # Uniformly sample redshift bins to ensure good coverage of redshift range
            nsample_per_bin = 500
            nbins = int(np.ceil(spec_sample_frac * len(data) / nsample_per_bin))
            redshiftbin = np.linspace(0.0, data["z"].max(), nbins)
            bin_ids = np.digitize(z, redshiftbin)
            RandomSample = []
            for b in range(1, len(redshiftbin)):
                idx = np.flatnonzero(bin_ids == b)
                if len(idx):
                    pick = np.random.choice(
                        idx,
                        size=min(nsample_per_bin, len(idx)),
                        replace=False,
                    )
                    RandomSample.extend(pick)
            specz = z[RandomSample]
            ids_specz = gal_ids[RandomSample]
            ra_specz = ra[RandomSample]
            dec_specz = dec[RandomSample]
            z_err = np.ones(np.shape(dec_specz)) * 0.00001
            table = Table()
            table["id"] = ids_specz
            table["ra"] = ra_specz
            table["dec"] = dec_specz
            table["z"] = specz
            table["z_err"] = z_err
            table.write(
                os.path.join(self.data_dir, f"SpeczSample_{file_suff}.fits"),
                format="fits",
                overwrite=True,
            )
            print(
                f"Wrote {os.path.join(self.data_dir, f'SpeczSample_{file_suff}.fits')} with {len(ra_specz)} galaxies."
            )
            del specz, ids_specz, ra_specz, dec_specz, z_err

        # Limiting fluxes for each band
        b_array_lsst = np.array([1.4e-12, 9.0e-13, 1.2e-12, 1.8e-12, 7.4e-12, 7.4e-12])
        b_array_roman = np.array([7.4e-12] * len(roman_bands))
        b_array = (
            np.concatenate([b_array_lsst, b_array_roman])
            if self.survey == "lsst_roman"
            else (b_array_lsst if self.survey == "lsst" else b_array_roman)
        )
        zp = 22.5

        influx = data[[f"flux_gold_{b}" for b in bands]].to_numpy()
        influx_err = data[[f"flux_err_gold_{b}" for b in bands]].to_numpy()

        with h5py.File(os.path.join(self.data_dir, diffsky_rm_file), "w") as f:
            f.create_dataset("redshift_true", data=z)
            f.create_dataset("id", data=gal_ids)
            f.create_dataset("ra", data=ra)
            f.create_dataset("dec", data=dec)
            for name, arr in zip(bands_inds, influx.T):
                f.create_dataset(f"flux_{name}", data=arr)
            for name, arr in zip(bands_inds, influx_err.T):
                f.create_dataset(f"fluxerr_{name}", data=arr)
            f.attrs["zeropoint"] = zp
            f.attrs["b_array"] = b_array
        print(
            f"Wrote {os.path.join(self.data_dir, diffsky_rm_file)} with {len(data)} galaxies."
        )

    def make_masks(self):
        def healpix_radec_rect_mask(
            nside, ra_min, ra_max, dec_min, dec_max, nest=False, dtype=np.uint8
        ):
            """
            Return a HEALPix mask (size hp.nside2npix(nside)) that is 1 inside
            the RA/Dec rectangle and 0 elsewhere.

            Angles in degrees. RA assumed in [0, 360).
            Handles RA wrap-around (e.g. ra_min=350, ra_max=10).
            """
            npix = hp.nside2npix(nside)
            ipix = np.arange(npix)

            theta, phi = hp.pix2ang(
                nside, ipix, nest=nest
            )  # theta: colat [0,pi], phi: lon [0,2pi)
            ra = np.degrees(phi)  # [0, 360)
            dec = 90.0 - np.degrees(theta)  # [-90, 90]

            # Dec cut
            in_dec = (dec >= dec_min) & (dec <= dec_max)

            # RA cut (with wrap support)
            ra_min = ra_min % 360.0
            ra_max = ra_max % 360.0
            # Special case: full sky in RA
            if np.isclose((ra_max - ra_min) % 360.0, 0.0):
                in_ra = np.ones_like(ra, dtype=bool)
            elif ra_min <= ra_max:
                in_ra = (ra >= ra_min) & (ra <= ra_max)
            else:
                # wrap-around: e.g. [350, 360) U [0, 10]
                in_ra = (ra >= ra_min) | (ra <= ra_max)

            mask = np.zeros(npix, dtype=dtype)
            mask[in_ra & in_dec] = 1
            return mask

        footprint_mask = healpix_radec_rect_mask(
            4096, 0.0, 300.0, 75.0, 85.0, nest=False
        )  # Sets where the survey observed data

        foreground_mask = np.zeros_like(
            footprint_mask
        )  # Should be 0 where data is good to use
        hp.write_map(
            os.path.join(self.data_dir, "footprint_mask.hpy"),
            footprint_mask,
            dtype=np.float32,
            overwrite=True,
        )
        hp.write_map(
            os.path.join(self.data_dir, "foreground_mask.hpy"),
            foreground_mask,
            dtype=np.float32,
            overwrite=True,
        )


if __name__ == "__main__":
    diffsky_rm = DiffSkyRedMapper(z_max=1.0, survey="lsst_roman")

    diffsky_rm.make_input_hdf5_with_flux_and_errs(make_specz_sample=True, spec_sample_frac=0.35)
    diffsky_rm.make_masks()

    # diffsky_rm.test_e2e_cat()
    # diffsky_rm.plot_ra_dec_distribution()
    # diffsky_rm.test_specz_sample()
