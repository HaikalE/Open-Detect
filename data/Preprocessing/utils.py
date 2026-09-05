import numpy as np
import binascii
import scapy.all as scapy


PACKETS_PER_FLOW = 8
HEADER_BYTES_PER_PACKET = 80
PAYLOAD_BYTES_PER_PACKET = 48
BYTES_PER_PACKET = HEADER_BYTES_PER_PACKET + PAYLOAD_BYTES_PER_PACKET
IMAGE_SIDE = 32
IMAGE_BYTES = IMAGE_SIDE * IMAGE_SIDE


# hex to 0-255
def string_to_hex_array(flow_string):
    return np.array([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)])


def read_pcap_list(pcap_filename, if_augment=False, remove_ip=True, keep_payload=True):
    """Convert eight packets into the paper's 32x32 grayscale input."""
    header_hex_length = HEADER_BYTES_PER_PACKET * 2
    payload_hex_length = PAYLOAD_BYTES_PER_PACKET * 2
    packets = scapy.rdpcap(pcap_filename)
    data = []
    flow_hex_length = IMAGE_BYTES * 2
    for packet in packets:
        try:
            header, payload = raw_packet_to_string(packet, remove_ip=remove_ip, keep_payload=keep_payload)
        except ValueError:
            # Excluded packets do not consume one of the first eight IP slots.
            continue
        data.append(header + payload)
        if not if_augment and len(data) == PACKETS_PER_FLOW:
            break

    if not data:
        return []

    if not if_augment or len(data) <= PACKETS_PER_FLOW:
        flow_string = ''.join(data)
        flow_string += '0' * (flow_hex_length - len(flow_string))
        flow_array = string_to_hex_array(flow_string)
        return [{
            "data": flow_array,
        }]
    else:
        assert len(data) > PACKETS_PER_FLOW
        flow_array_list = []
        for i in range(len(data) - PACKETS_PER_FLOW + 1):
            flow_string = ''.join(data[i:i + PACKETS_PER_FLOW])
            flow_array_list.append(string_to_hex_array(flow_string))
        return [{
            "data": flow_array,
        } for flow_array in flow_array_list]


def raw_packet_to_string(packet, remove_ip=True, keep_payload=True):
    """Keep network/transport headers and application bytes as separate regions.

    Read the transport payload structurally, including decoded application layers;
    a Scapy Raw layer is not required. Never modify the caller's captured packet.
    """
    header_hex_length = HEADER_BYTES_PER_PACKET * 2
    payload_hex_length = PAYLOAD_BYTES_PER_PACKET * 2
    if scapy.IP in packet:
        ip = packet[scapy.IP].copy()
        if ip.frag or ip.flags.MF:
            raise ValueError('Fragmented packets require reassembly before extraction')
        pad_address = '0.0.0.0'
    elif scapy.IPv6 in packet:
        ip = packet[scapy.IPv6].copy()
        if scapy.IPv6ExtHdrFragment in ip:
            raise ValueError('Fragmented packets require reassembly before extraction')
        pad_address = '::'
    else:
        raise ValueError('Non-IP packet')
    if scapy.TCP in ip:
        transport = ip[scapy.TCP]
    elif scapy.UDP in ip:
        transport = ip[scapy.UDP]
        if transport.sport in (67, 68, 546, 547) or transport.dport in (67, 68, 546, 547):
            raise ValueError('DHCP excluded')
    else:
        raise ValueError('Expected a TCP or UDP flow')
    # Remove only capture padding, not application data or transport options.
    if scapy.Padding in ip:
        ip[scapy.Padding].underlayer.remove_payload()
    application_bytes = bytes(transport.payload)
    if remove_ip:
        ip.src, ip.dst = pad_address, pad_address
    network_bytes = bytes(ip)
    header_length = len(network_bytes) - len(application_bytes)
    header = network_bytes[:header_length].hex()
    payload = application_bytes.hex() if keep_payload else ''
    header = (
        header[:header_hex_length]
        if len(header) > header_hex_length
        else header + '0' * (header_hex_length - len(header))
    )
    payload = (
        payload[:payload_hex_length]
        if len(payload) > payload_hex_length
        else payload + '0' * (payload_hex_length - len(payload))
    )
    return header, payload



if __name__ == "__main__":
    pass

