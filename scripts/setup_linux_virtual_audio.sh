#!/usr/bin/env bash
set -euo pipefail

sink_name="${WITCH_VIRTUAL_AUDIO_SINK:-WitchVirtualCable}"
description="${WITCH_VIRTUAL_AUDIO_DESCRIPTION:-WitchVirtualCable}"
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire-pulse.conf.d"
config_file="$config_dir/10-witch-virtual-cable.conf"

mkdir -p "$config_dir"
cat > "$config_file" <<EOF
pulse.cmd = [
    {
        cmd = "load-module"
        args = "module-null-sink sink_name=$sink_name sink_properties=device.description=$description channels=2 rate=48000"
        flags = [ ]
    }
]
EOF

if pactl list short sinks | awk '{print $2}' | grep -Fxq "$sink_name"; then
    echo "Virtual audio sink already active: $sink_name"
else
    pactl load-module module-null-sink \
        sink_name="$sink_name" \
        sink_properties=device.description="$description" \
        channels=2 \
        rate=48000 >/dev/null
    echo "Loaded virtual audio sink: $sink_name"
fi

echo "Persistent PipeWire config: $config_file"
pactl list short sinks | grep -E "(^|[[:space:]])${sink_name}([[:space:]]|$)" || true
pactl list short sources | grep -E "(^|[[:space:]])${sink_name}\\.monitor([[:space:]]|$)" || true
