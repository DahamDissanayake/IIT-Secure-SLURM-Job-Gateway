# bootstrap-install.md — One-time install of the canonical clone

> Run once on the **login node** as an admin. Establishes `/opt/slurm-deck` as the
> single live checkout that every `gpuusers` member runs via the launcher.

```bash
# 1. Canonical live clone — readable by all gpuusers, pull-to-update by admin only
sudo git clone https://github.com/DahamDissanayake/slurm-deck.git /opt/slurm-deck
sudo chown -R slurmadmin:gpuusers /opt/slurm-deck
sudo chmod -R 0750 /opt/slurm-deck

# 2. Site configuration (git-ignored)
sudo -u slurmadmin cp /opt/slurm-deck/deploy/site.env.example /opt/slurm-deck/deploy/site.env
sudo -u slurmadmin nano /opt/slurm-deck/deploy/site.env     # edit for your cluster

# 3. Launcher — the single integration point (PYTHONPATH points at the clone)
sudo tee /usr/local/bin/slurm-deck >/dev/null <<'LAUNCHER'
#!/bin/bash
exec env -i \
    HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
    PATH="/shared/miniforge3/bin:/usr/local/bin:/usr/bin:/bin" \
    SSH_CLIENT="${SSH_CLIENT:-}" TERM="${TERM:-xterm}" \
    PYTHONPATH="/opt/slurm-deck" \
    SD_SITE_ENV="/opt/slurm-deck/deploy/site.env" \
    /usr/bin/python3 -m slurmdeck
LAUNCHER
sudo chmod 0755 /usr/local/bin/slurm-deck

# 4. Forced TUI for the gateway group (sshd)
#    /etc/ssh/sshd_config.d/slurm-deck-gateway.conf:
#      Match Group gpuusers
#          ForceCommand /usr/local/bin/slurm-deck
#          (+ the no-forwarding hardening from M02 §9)
#    Adding a user to gpuusers is then all it takes to grant the tool.

# 5. Update for everyone, any time:
cd /opt/slurm-deck && git pull --ff-only && python3 -m pytest tests/ -q
```
