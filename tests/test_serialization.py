from src import stan_runner
from src.stan_runner import *

from hatchet_sdk import Hatchet, WorkflowList
from hatchet_sdk.client import ClientImpl

hatchet: Hatchet = get_hatchet()


def test():
    from hatchet_sdk.client import new_client
    client:ClientImpl = new_client()
    admin = client.admin
    # l = client.rest_client.workflow_api.workflow_list(client.rest_client.tenant_id)
    client.rest_client.tenant_id
    rest_client = client.rest_client
    wl = rest_client.workflow_list()
    rest_client.workflow_run_list(workflow_id="Hatchet_StanRunner")
    # list: WorkflowList = hatchet.admin.list_workflows(client.rest_client.tenant_id)
    print(list)


if __name__ == '__main__':
    test()
