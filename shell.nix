# Nix-shell fallback for environments without flakes enabled.
#
# For the PINNED, reproducible environment prefer the flake:  nix develop
# (that pins nixpkgs via flake.lock). This file resolves nixpkgs from the
# caller's <nixpkgs> channel.
#
# It re-exports the flake's devShell when flakes are available, else builds an
# equivalent shell from the channel.
{ pkgs ? import <nixpkgs> { } }:

let
  lib = pkgs.lib;
  python = pkgs.python3.withPackages (ps: with ps; [ numpy pyyaml pytest xxhash ]);
  common = with pkgs; [
    cmake gnumake ninja pkg-config
    libsndfile
    llvmPackages.openmp        # libomp
    llvm                       # llvm-mca + llvm-profdata
    qemu                       # qemu-system-arm
    gcc-arm-embedded           # arm-none-eabi-gcc (+newlib)
    ripgrep dfu-util git
    python
  ];
  linuxOnly = lib.optionals pkgs.stdenv.isLinux (with pkgs; [ fftw ]);
  fw = pkgs.darwin.apple_sdk.frameworks or { };
  darwinFrameworks = lib.optionals pkgs.stdenv.isDarwin (
    builtins.filter (x: x != null) (map (n: fw.${n} or null) [ "Accelerate" "Metal" "Foundation" ])
  );
in
pkgs.mkShell {
  packages = common ++ linuxOnly ++ darwinFrameworks;
  shellHook = ''
    echo "Spectral Engine dev shell (shell.nix; prefer 'nix develop' for the pinned env)"
  '';
}
