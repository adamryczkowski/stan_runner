import datetime
import socket
import time

import humanize
import netifaces
from adams_text_utils import format_txt_list
from overrides import overrides

from .nats_DTO import SerializableObjectInfo


class BrokerInfo(SerializableObjectInfo):
    _hostname: str
    _network_addresses: dict[str, list[str]]  # All the network interfaces and their addresses

    @staticmethod
    def CreateFromLocalHost():
        hostname = socket.gethostname()

        network_addresses = {}
        for iface in netifaces.interfaces():
            iface_addresses = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in iface_addresses:
                for addr in iface_addresses[netifaces.AF_INET]:
                    if addr["addr"] == "127.0.0.1":
                        continue
                    if addr["addr"][0:4] == "172.":
                        continue
                    if iface not in network_addresses:
                        network_addresses[iface] = []
                    network_addresses[iface].append(addr["addr"])

        if "default" in netifaces.gateways():
            default_gateway = netifaces.gateways()["default"]
            if netifaces.AF_INET in default_gateway:
                gateway_ip, iface = default_gateway[netifaces.AF_INET]
                default_entry = network_addresses[iface]
                # Put the default gateway first
                del network_addresses[iface]
                network_addresses = {iface: default_entry, **network_addresses}

        return BrokerInfo(hostname, network_addresses)

    def __init__(self, hostname: str, network_addresses: dict[str, list[str]], object_id: str = None,
                 timestamp: float = None):
        super().__init__(object_id)
        self._hostname = hostname
        self._network_addresses = network_addresses

    @overrides
    def pretty_print(self)->str:
        if self.timestamp is None:
            ans = f"Broker {self.object_id} \"{self._hostname}\", never seen"
        else:
            ans = f"Broker {self.object_id} \"{self._hostname}\", last seen {humanize.naturaltime(datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(self.timestamp))}"

        ans += f""", with network addresses:\n\n"""

        ifaces = []
        for iface, addresses in self._network_addresses.items():
            ifaces.append(f"* {iface}: {format_txt_list(addresses, max_length=5)}")

        return ans + "\n".join(ifaces)

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["hostname"] = self._hostname
        d["network_addresses"] = self._network_addresses
        return d

    @overrides
    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._hostname = state["hostname"]
        self._network_addresses = state["network_addresses"]

    @property
    def hostname(self):
        return self._hostname

    @property
    def network_addresses(self):
        return self._network_addresses



