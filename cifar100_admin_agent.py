from stadle import AdminAgent
from stadle.lib.util import client_arg_parser
from stadle.lib.entity.model import BaseModel
from stadle import BaseModelConvFormat

# import model architecture
from models.samplenet import SampleNet

def get_samplenet_model():
    return BaseModel("PyTorch-CIFAR100-Model", SampleNet(), BaseModelConvFormat.pytorch_format)

if __name__ == '__main__':
    args = client_arg_parser()

    admin_agent = AdminAgent(config_file="config/config_agent.json", simulation_flag=args.simulation,
                             aggregator_ip_address=args.aggregator_ip, reg_port=args.reg_port, agent_name=args.agent_name,
                             exch_port=args.exch_port, model_path=args.model_path, base_model=get_samplenet_model(),
                             agent_running=False)

    admin_agent.preload()
    admin_agent.initialize()