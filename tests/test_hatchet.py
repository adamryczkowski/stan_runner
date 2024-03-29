from dotenv import load_dotenv
from stan_runner import *

def test1():
    load_dotenv()


    # Create a Hatchet instance that will be shared across all workflows

    worker = hatchet.worker('example-worker')
    worker.register_workflow(Hatchet_StanRunner())
    worker.start()

if __name__ == '__main__':
    test1()
