import os
REINFLOW_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not REINFLOW_DIR == os.environ['REINFLOW_DIR']:
    raise ValueError(f"Hey did you correctly set up your env variable REINFLOW_DIR? It showws that REINFLOW_DIR={os.environ['REINFLOW_DIR']} but the code rest in {REINFLOW_DIR}. ")

REINFLOW_CFG_DIR = os.path.join(REINFLOW_DIR, 'cfg')

REINFLOW_DATA_DIR = os.environ['REINFLOW_DATA_DIR'] #os.path.join(REINFLOW_DIR, 'data')

REINFLOW_LOG_DIR =  os.environ['REINFLOW_LOG_DIR'] #os.path.join(REINFLOW_DIR, 'log')
