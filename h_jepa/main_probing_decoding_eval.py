import hydra

from final_probing_decoding_eval import run_final_probing_decoding_eval


@hydra.main(version_base=None, config_path="config/probing", config_name=None)
def run(cfg):
    run_final_probing_decoding_eval(cfg)


if __name__ == "__main__":
    run()
