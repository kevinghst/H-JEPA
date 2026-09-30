#!/usr/bin/env bash
# Prints (does not run) the commands that move runs of the old layout into ckpts/<env>/:
# plain runs ckpts/<env>_<model>*/ and launch.py sweep dirs ckpts/<sweep>_<ts>/ (env from the
# launch.json config_name), and that prefix each saved subdir with <env>/ so main_hjepa.py and
# `launch.py resume` find the moved run dir. Review the output, then pipe it to bash.
#   scripts/migrate_ckpts_layout.sh [ckpts_dir, default $STABLEWM_HOME/ckpts]
set -e
CKPTS=${1:-$STABLEWM_HOME/ckpts}

for d in "$CKPTS"/*/; do
  d=${d%/} name=$(basename "$d")
  env=
  if [[ -f $d/launch.json ]]; then
    env=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))[0].get("config_name", "").split("_")[0])' "$d/launch.json")
  elif [[ $name =~ ^(ant|fourroom|cube|pusht|droid)_(lewm|hjepa_l[0-9]|hwm_l[0-9]) ]]; then
    env=${BASH_REMATCH[1]}
  fi
  case $env in ant | fourroom | cube | pusht | droid) ;; *) echo "# skip $d" && continue ;; esac
  echo "mkdir -p $CKPTS/$env && mv $d $CKPTS/$env/"
  echo "find $CKPTS/$env/$name -maxdepth 3 -regex '.*/seed[0-9]*/config\.yaml' -exec sed -i 's|^subdir: |subdir: $env/|' {} +"
done
