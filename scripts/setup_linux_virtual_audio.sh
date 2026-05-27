#!/usr/bin/env bash
set -euo pipefail

sink_name="${WITCH_VIRTUAL_AUDIO_SINK:-VirtualCable}"
description="${WITCH_VIRTUAL_AUDIO_DESCRIPTION:-VirtualCable}"
source_name="${WITCH_VIRTUAL_AUDIO_SOURCE:-$sink_name}"
source_description="${WITCH_VIRTUAL_AUDIO_SOURCE_DESCRIPTION:-$description}"
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire-pulse.conf.d"
config_file="$config_dir/10-virtual-cable.conf"
legacy_config_file="$config_dir/10-witch-virtual-cable.conf"

mkdir -p "$config_dir"
if [ -f "$legacy_config_file" ] && [ "$legacy_config_file" != "$config_file" ]; then
    mv "$legacy_config_file" "$config_file"
fi

cat > "$config_file" <<EOF
pulse.cmd = [
    {
        cmd = "load-module"
        args = "module-null-sink sink_name=$sink_name sink_properties=device.description=$description channels=2 rate=48000"
        flags = [ ]
    },
    {
        cmd = "load-module"
        args = "module-remap-source master=$sink_name.monitor source_name=${source_name}-input source_properties=device.description=$source_description channels=2 rate=48000 remix=no"
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

source_full_name="${source_name}-input"
if pactl list short sources | awk '{print $2}' | grep -Fxq "$source_full_name"; then
    echo "Virtual audio source already active: $source_full_name"
else
    pactl load-module module-remap-source \
        master="$sink_name.monitor" \
        source_name="$source_full_name" \
        source_properties="device.description=$source_description" \
        channels=2 \
        rate=48000 \
        remix=no >/dev/null
    echo "Loaded virtual audio source: $source_full_name"
fi

echo "Persistent PipeWire config: $config_file"
pactl list short sinks | grep -E "(^|[[:space:]])${sink_name}([[:space:]]|$)" || true
pactl list short sources | grep -E "(^|[[:space:]])${sink_name}\\.monitor([[:space:]]|$)" || true
pactl list short sources | grep -E "(^|[[:space:]])${source_full_name}([[:space:]]|$)" || true
