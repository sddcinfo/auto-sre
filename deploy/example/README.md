# Deployment examples

Reference configs for running auto-sre as a system service. These files
are **templates** — they contain `${PLACEHOLDER}` variables you must fill
in for your environment before installing them.

## systemd

`systemd/autosre.service.template` runs `autosre start` as a system service
on every boot. Render the template with your own values via `envsubst`:

```sh
export SERVICE_USER=$(id -un)                  # the account autosre runs as
export SERVICE_HOME=$(getent passwd "$SERVICE_USER" | cut -d: -f6)
export REPO_DIR=$(pwd)                          # checkout of auto-sre
export PYTHON_BIN_DIR=$(dirname "$(command -v python3)")
export NODE_BIN_DIR=$(dirname "$(command -v node 2>/dev/null || echo /usr/bin/true)")

envsubst '${SERVICE_USER} ${SERVICE_HOME} ${REPO_DIR} ${PYTHON_BIN_DIR} ${NODE_BIN_DIR}' \
  < deploy/example/systemd/autosre.service.template \
  | sudo tee /etc/systemd/system/autosre.service >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now autosre.service
```

The placeholders expand to the resolved paths at install time. The
template is never edited in place, so future updates to the repo don't
clobber your local customisations.

## Dropbox

The dropbox subsystem ships a first-class installer — there's no
template to render by hand. Use the CLI instead:

```sh
cp deploy/example/dropbox.toml ~/.config/autosre/dropbox.toml
$EDITOR ~/.config/autosre/dropbox.toml      # set data_dir + ports to taste

autosre dropbox install --config-file ~/.config/autosre/dropbox.toml
autosre dropbox init    --config-file ~/.config/autosre/dropbox.toml --password-stdin
autosre dropbox start
```

See the main README for the full `autosre dropbox` command reference.
