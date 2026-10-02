"""Paper runs behind each release training config (seed placeholder {s}), relative to $HJEPA_HOME/ckpts."""
import os
import re

ROOT = os.path.join(os.environ["HJEPA_HOME"], "ckpts") + "/"
R = {
    "ant": {"lewm": "ant/7-28-2/ant_level1_union_ds_seed{s}__stage1", "hjepa_l2": "ant/9-6-3/1/seed{s}",
            "hjepa_l3": "ant/9-6-1/1/seed{s}", "hjepa_l4": "ant/9-6-2/1/seed{s}", "hwm_l2": "ant/9-12-5/0/seed{s}",
            "hwm_l3": "ant/9-12-6/0/seed{s}", "hwm_l4": "ant/9-12-7/0/seed{s}"},
    "fourroom": {"lewm": "fourroom_distractors/7-23-2/0/seed{s}", "hjepa_l2": "fourroom_distractors/9-11-1/4/seed{s}",
                 "hjepa_l3": "fourroom_distractors/9-11-2/4/seed{s}", "hjepa_l4": "fourroom_distractors/9-11-3/4/seed{s}",
                 "hwm_l2": "fourroom_distractors/9-12-5/0/seed{s}", "hwm_l3": "fourroom_distractors/9-12-6/0/seed{s}",
                 "hwm_l4": "fourroom_distractors/9-12-7/0/seed{s}"},
    "cube": {"lewm": "ogb/7-28-1/ogb_level1_seed{s}__stage1", "hjepa_l2": "ogb/9-10-1/0/seed{s}",
             "hjepa_l3": "ogb/9-10-2/0/seed{s}", "hjepa_l4": "ogb/9-12-8/0/seed{s}", "hwm_l2": "ogb/9-12-5/0/seed{s}",
             "hwm_l3": "ogb/9-12-6/0/seed{s}", "hwm_l4": "ogb/9-12-7/0/seed{s}"},
    "pusht": {"lewm": "pusht/7-28-1/pusht_level1_seed{s}__stage1", "hjepa_l2": "pusht/9-9-1/0/seed{s}",
              "hjepa_l3": "pusht/9-9-2/0/seed{s}", "hjepa_l4": "pusht/9-9-3/0/seed{s}", "hwm_l2": "pusht/9-12-5/0/seed{s}",
              "hwm_l3": "pusht/9-12-6/0/seed{s}", "hwm_l4": "pusht/9-12-7/0/seed{s}"},
}


def checkpoint(env, model, seed):
    run = ROOT + R[env][model].format(s=seed)
    name = os.path.basename(run) if model == "lewm" and env != "fourroom" else "model"
    return f"{run}/{name}_object.ckpt"


def rename_datasets(cfg):
    # The release renamed the paper datasets (dropped `_2_5x` and the FourRoom `fourroom_7_21/tp35/` dirs).
    return {k: re.sub(r"visual_antmaze_medium_(stitch_val|probing_train|probing_eval)_2_5x", r"visual_antmaze_medium_\1",
                      v).replace("fourroom_7_21/tp35/", "") if isinstance(v, str) else v
            for k, v in cfg.items()}
