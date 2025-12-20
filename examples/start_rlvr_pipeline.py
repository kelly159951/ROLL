import argparse
import os

from dacite import from_dict, Config
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline

# import debugpy
# debugpy.listen(2332)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app") #初始化配置管理系统
    cfg = compose(config_name=args.config_name) #加载配置文件

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config: RLVRConfig = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True)) #将配置对象转换成数据类实例

    init()
    pipeline = RLVRPipeline(pipeline_config=ppo_config) #创建RLVR训练流水线实例

    pipeline.run()


if __name__ == "__main__":
    main()
