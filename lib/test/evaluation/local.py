from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ""
    settings.got10k_lmdb_path = ""
    settings.got10k_path = "/data/GOT-10k"
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.lasot_lmdb_path = ""
    settings.lasot_path = "/home/guest/XieBailian/proj/stark_lt/data/lasot"
    settings.network_path = "/home/guest/XieBailian/proj/stark_lt/test/networks"  # Where tracking networks are stored.
    settings.nfs_path = ""
    settings.otb_path = ""
    settings.prj_dir = "/home/guest/XieBailian/proj/stark_lt"
    settings.result_plot_path = "/home/guest/XieBailian/proj/stark_lt/test/result_plots"
    settings.results_path = "/home/guest/XieBailian/proj/stark_lt/test/tracking_results"  # Where to store tracking results
    settings.save_dir = "/home/guest/XieBailian/proj/stark_lt"
    settings.segmentation_path = (
        "/home/guest/XieBailian/proj/stark_lt/test/segmentation_results"
    )
    settings.tc128_path = ""
    settings.tn_packed_results_path = ""
    settings.tpl_path = ""
    settings.trackingnet_path = ""
    settings.uav_path = ""
    settings.vot_path = "/data/VOT2020_LT"
    settings.youtubevos_dir = ""

    return settings
