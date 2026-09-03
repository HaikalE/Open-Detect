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
    end = len(packets) if if_augment else PACKETS_PER_FLOW
    for packet in packets[:end]:
        try:
            header, payload = raw_packet_to_string(packet, remove_ip=remove_ip, keep_payload=keep_payload)
        except Exception:
            # Non-IP or malformed packets still occupy one zero-padded slot.
            header = '0' * header_hex_length
            payload = '0' * payload_hex_length
        data.append(header + payload)

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
    """Keep 80 header bytes and 48 payload bytes from one packet."""
    header_hex_length = HEADER_BYTES_PER_PACKET * 2
    payload_hex_length = PAYLOAD_BYTES_PER_PACKET * 2
    ip = packet["IP"]
    if remove_ip:
        PAD_IP_ADDR = "0.0.0.0"
        ip.src, ip.dst = PAD_IP_ADDR, PAD_IP_ADDR
    header = (binascii.hexlify(bytes(ip))).decode()
    if keep_payload:
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
    else:
        payload = ''
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

