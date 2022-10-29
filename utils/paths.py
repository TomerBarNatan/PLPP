from pathlib import Path

multiSiteMri_int_to_site = {0: 'ISBI', 1: "ISBI_1.5", 2: 'I2CVB', 3: "UCL", 4: "BIDMC", 5: "HK"}
multiSiteMri_site_to_int = {v: k for k, v in multiSiteMri_int_to_site.items()}
cc359_data_path = '/CC359'
cc359_splits_dir = Path('/CC359_splits/')
cc359_results = Path('/CC359_results/')
msm_data_path = '/multiSiteMRI'
msm_splits_dir = Path('/msm_splits')
msm_results = Path('/msm_results')

