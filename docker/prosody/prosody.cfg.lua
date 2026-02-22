-- Minimal Prosody config for local SPADE testing
admins = { "admin@localhost" }
modules_enabled = {
    "roster";
    "saslauth";
    "tls";
    "dialback";
    "disco";
    "posix";
    "ping";
    "register";
    "admin_adhoc";
}
modules_disabled = {
    "s2s";
}
allow_registration = true
c2s_require_encryption = false
ssl = {
    key = "/var/lib/prosody/localhost.key";
    certificate = "/var/lib/prosody/localhost.crt";
}
c2s_ports = { 5222 }
authentication = "internal_plain"

VirtualHost "localhost"
    allow_registration = true
