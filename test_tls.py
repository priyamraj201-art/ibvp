"""
Unit tests for the self-signed TLS certificate generator used to serve the
dashboard over HTTPS (required for phone camera access, which browsers
only permit on a secure context).
"""

import os
import sys

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from dashboard.tls import ensure_self_signed_cert


def test_generates_cert_and_key_files(tmp_path):
    cert_path, key_path = ensure_self_signed_cert(str(tmp_path), "192.168.1.50")

    assert os.path.isfile(cert_path)
    assert os.path.isfile(key_path)


def test_cert_includes_lan_ip_and_localhost_in_san():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cert_path, _ = ensure_self_signed_cert(d, "192.168.1.50")

        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        dns_names = san.get_values_for_type(x509.DNSName)
        ip_addresses = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]

        assert "localhost" in dns_names
        assert "127.0.0.1" in ip_addresses
        assert "192.168.1.50" in ip_addresses


def test_key_is_loadable_without_password():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        _, key_path = ensure_self_signed_cert(d, "10.0.0.5")

        with open(key_path, "rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=None)

        assert key.key_size == 2048


def test_invalid_ip_is_skipped_without_raising():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cert_path, key_path = ensure_self_signed_cert(d, "not-an-ip")

        assert os.path.isfile(cert_path)
        assert os.path.isfile(key_path)


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
