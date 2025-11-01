from imitation_learning.agents.bc import BC_CONFIG_DICT, BCAgent, BCMSEAgent
from imitation_learning.agents.gcbc import GCBC_CONFIG_DICT, GCBCAgent, GCBCMSEAgent
from imitation_learning.agents.gciql import GCIQL_CONFIG_DICT, GCIQLAgent
from imitation_learning.agents.hiql import HIQL_CONFIG_DICT, HIQLAgent
from imitation_learning.agents.iql import IQL_CONFIG_DICT, IQLAgent

agents = {
    "gcbc": (GCBCAgent, GCBC_CONFIG_DICT),
    "gcbc_mse": (GCBCMSEAgent, GCBC_CONFIG_DICT),
    "gciql": (GCIQLAgent, GCIQL_CONFIG_DICT),
    "hiql": (HIQLAgent, HIQL_CONFIG_DICT),
    "bc": (BCAgent, BC_CONFIG_DICT),
    "bc_mse": (BCMSEAgent, BC_CONFIG_DICT),
    "iql": (IQLAgent, IQL_CONFIG_DICT)
}
