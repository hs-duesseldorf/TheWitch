#!/usr/bin/env bash
set -euo pipefail

sink_name="${WITCH_VIRTUAL_AUDIO_SINK:-VirtualCable}"
source_name="${WITCH_VIRTUAL_AUDIO_SOURCE:-$sink_name}"
source_full_name="${source_name}-input"
base_description="${WITCH_VIRTUAL_AUDIO_DESCRIPTION:-VirtualCable}"
sink_description="${WITCH_VIRTUAL_AUDIO_SINK_DESCRIPTION:-${base_description} Output}"
source_description="${WITCH_VIRTUAL_AUDIO_SOURCE_DESCRIPTION:-${base_description} Input}"
rate="${WITCH_VIRTUAL_AUDIO_RATE:-48000}"
channels="${WITCH_VIRTUAL_AUDIO_CHANNELS:-2}"

pulse_config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire-pulse.conf.d"
pulse_config_file="$pulse_config_dir/10-virtual-cable.conf"
legacy_pulse_config_file="$pulse_config_dir/10-witch-virtual-cable.conf"
pipewire_config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire.conf.d"
pipewire_config_file="$pipewire_config_dir/10-virtual-cable.conf"
legacy_pipewire_config_file="$pipewire_config_dir/10-witch-virtual-cable.conf"

have_command() {
    command -v "$1" >/dev/null 2>&1
}

write_pulse_config() {
    mkdir -p "$pulse_config_dir"
    if [ -f "$legacy_pulse_config_file" ] && [ "$legacy_pulse_config_file" != "$pulse_config_file" ]; then
        mv "$legacy_pulse_config_file" "$pulse_config_file"
    fi

    cat > "$pulse_config_file" <<EOF_PULSE
pulse.cmd = [
    {
        cmd = "load-module"
        args = "module-null-sink sink_name=$sink_name sink_properties=device.description=\"$sink_description\" channels=$channels rate=$rate"
        flags = [ ]
    },
    {
        cmd = "load-module"
        args = "module-remap-source master=$sink_name.monitor source_name=$source_full_name source_properties=device.description=\"$source_description\" channels=$channels rate=$rate remix=no"
        flags = [ ]
    }
]
EOF_PULSE
}

remove_pipewire_fallback_config() {
    for file in "$pipewire_config_file" "$legacy_pipewire_config_file"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo "Removed duplicate PipeWire fallback config: $file"
        fi
    done
}

setup_with_pactl() {
    write_pulse_config
    remove_pipewire_fallback_config

    if pactl list short sinks | awk '{print $2}' | grep -Fxq "$sink_name"; then
        echo "Virtual audio sink already active: $sink_name"
    else
        pactl load-module module-null-sink \
            sink_name="$sink_name" \
            sink_properties=device.description="$sink_description" \
            channels="$channels" \
            rate="$rate" >/dev/null
        echo "Loaded virtual audio sink: $sink_name"
    fi

    if pactl list short sources | awk '{print $2}' | grep -Fxq "$source_full_name"; then
        echo "Virtual audio source already active: $source_full_name"
    else
        pactl load-module module-remap-source \
            master="$sink_name.monitor" \
            source_name="$source_full_name" \
            source_properties="device.description=$source_description" \
            channels="$channels" \
            rate="$rate" \
            remix=no >/dev/null
        echo "Loaded virtual audio source: $source_full_name"
    fi

    echo "Persistent PipeWire Pulse config: $pulse_config_file"
    echo "Output device: $sink_name ($sink_description)"
    echo "Input device:  $source_full_name ($source_description)"
    echo "Note: some mixers also show ${sink_name}.monitor; use $source_full_name as the virtual microphone/input."
}
pipewire_sink_exists() {
    pw-cli ls Node |
        awk -v name="$sink_name" '$1 == "node.name" && $3 == "\"" name "\"" { found=1 } END { exit !found }'
}

setup_with_pipewire() {
    write_pulse_config
    remove_pipewire_fallback_config

    if pipewire_sink_exists; then
        echo "Virtual audio sink already active: $sink_name"
    else
        pw-cli create-node adapter "{ factory.name=support.null-audio-sink node.name=$sink_name node.description=\"$sink_description\" media.class=Audio/Sink object.linger=true audio.position=[ FL FR ] }" >/dev/null
        echo "Loaded virtual audio sink: $sink_name"
    fi

    echo "Persistent PipeWire Pulse config: $pulse_config_file"
    echo "pactl was not found, so the named input source cannot be loaded immediately from this shell."
    echo "Restart PipeWire/PipeWire Pulse or install pulseaudio-utils and rerun this script to activate: $source_full_name ($source_description)"
    echo "Output device: $sink_name ($sink_description)"
    echo "If your mixer shows ${sink_name}.monitor, that is the raw monitor for the output sink; the intended virtual microphone/input is $source_full_name."
    pw-cli ls Node | awk -v name="$sink_name" '
        /^\tid / { id=$2; sub(/,/, "", id) }
        $1 == "node.name" && $3 == "\"" name "\"" { print "PipeWire node id: " id; print $0 }
    '
}

if have_command pactl; then
    setup_with_pactl
elif have_command pw-cli; then
    setup_with_pipewire
else
    echo "Neither pactl nor pw-cli is installed." >&2
    echo "On Ubuntu, install one of these packages and rerun:" >&2
    echo "  sudo apt install pulseaudio-utils" >&2
    echo "  sudo apt install pipewire-bin" >&2
    exit 1
fi
