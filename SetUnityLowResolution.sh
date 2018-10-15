#!/bin/bash

mkdir -p ~/.config/unity3d/DefaultCompany/DonkeySim

cat > ~/.config/unity3d/DefaultCompany/DonkeySim/prefs << EOF
<unity_prefs version_major="1" version_minor="1">
        <pref name="Screenmanager Fullscreen mode" type="int">3</pref>
        <pref name="Screenmanager Resolution Height" type="int">480</pref>
        <pref name="Screenmanager Resolution Use Native" type="int">0</pref>
        <pref name="Screenmanager Resolution Width" type="int">640</pref>
</unity_prefs>
EOF
