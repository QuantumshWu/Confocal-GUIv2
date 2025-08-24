from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime, inspect, os

CERTIFICATE_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getsourcefile(inspect.currentframe()))), 'certificate')


def generate_ssl_certificate():
    os.makedirs(CERTIFICATE_DIR, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    key_path = os.path.join(CERTIFICATE_DIR, "server.key")
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
        )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME,            "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME,   "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME,            "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME,        "Organization"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "OrgUnit"),
        x509.NameAttribute(NameOID.COMMON_NAME,              "my-rf-server"),
    ])

    cert = (
        x509.CertificateBuilder()
           .subject_name(subject)
           .issuer_name(issuer)
           .public_key(key.public_key())
           .serial_number(x509.random_serial_number())
           .not_valid_before(datetime.datetime.utcnow())
           .not_valid_after(
               datetime.datetime.utcnow() + datetime.timedelta(days=3650)
           )
           .add_extension(
               x509.SubjectAlternativeName([x509.DNSName("my-rf-server")]),
               critical=False
           )
           .sign(key, hashes.SHA256())
    )

    cert_path = os.path.join(CERTIFICATE_DIR, "server.crt")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Generated:\n  {key_path}\n  {cert_path}\nValidity: 10 years")