"""
Self-signed TLS certificate generation for the dashboard's local HTTPS server.

Browsers only grant camera access (getUserMedia) on a "secure context" —
HTTPS or localhost. A plain http://<lan-ip> origin does not qualify, even
though the phone camera feature only ever operates on the local network.
Serving over HTTPS with a locally-generated, self-signed certificate keeps
the feature LAN-only (no external CA, no internet dependency) while
satisfying that browser requirement. The phone's browser will show a
one-time "connection isn't private" warning the first time it connects,
which is expected and safe to accept for a local development/LAN tool.
"""

import datetime
import ipaddress
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def ensure_self_signed_cert(cert_dir: str, lan_ip: str) -> tuple[str, str]:
    """Generate a fresh self-signed certificate covering localhost and the
    given LAN IP, writing it to cert_dir. Regenerated on every call since
    the LAN IP can change between network connections."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "dashboard_cert.pem")
    key_path = os.path.join(cert_dir, "dashboard_key.pem")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "IBVAP Surveillance Dashboard"),
    ])

    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except ValueError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    return cert_path, key_path
