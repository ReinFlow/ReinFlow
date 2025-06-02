# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
REINFLOW_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not REINFLOW_DIR == os.environ['REINFLOW_DIR']:
    raise ValueError(f"Hey did you correctly set up your env variable REINFLOW_DIR? It showws that REINFLOW_DIR={os.environ['REINFLOW_DIR']} but the code rest in {REINFLOW_DIR}. ")

REINFLOW_CFG_DIR = os.path.join(REINFLOW_DIR, 'cfg')

REINFLOW_DATA_DIR = os.environ['REINFLOW_DATA_DIR'] #os.path.join(REINFLOW_DIR, 'data')

REINFLOW_LOG_DIR =  os.environ['REINFLOW_LOG_DIR'] #os.path.join(REINFLOW_DIR, 'log')
