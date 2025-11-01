from imitation_learning.agents.gcbc import GCBCAgent, GCBCMSEAgent, GCBC_CONFIG_DICT
from imitation_learning.agents.gciql import GCIQLAgent, GCIQL_CONFIG_DICT
from imitation_learning.agents.hiql import HIQLAgent, HIQL_CONFIG_DICT
from imitation_learning.agents.bc import BCAgent, BCMSEAgent, BC_CONFIG_DICT
from imitation_learning.agents.iql import IQLAgent, IQL_CONFIG_DICT

agents = {
    "gcbc": (GCBCAgent, GCBC_CONFIG_DICT),
    "gcbc_mse": (GCBCMSEAgent, GCBC_CONFIG_DICT),
    "gciql": (GCIQLAgent, GCIQL_CONFIG_DICT),
    "hiql": (HIQLAgent, HIQL_CONFIG_DICT),
    "bc": (BCAgent, BC_CONFIG_DICT),
    "bc_mse": (BCMSEAgent, BC_CONFIG_DICT),
    "iql": (IQLAgent, IQL_CONFIG_DICT)
}