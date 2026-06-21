{
  # Spectral Engine — reproducible dev/build environment.
  #
  # Pins EVERY installable dependency (see README "DEPENDENCIES") so a build is
  # bit-reproducible across machines: the desktop toolchain (cmake, compiler,
  # OpenMP, libsndfile, FFTW), the GPU/accel frameworks available on the host,
  # and the embedded ARM performance stack (qemu-system-arm + llvm-mca + an
  # arm-none-eabi cross-gcc) that the M7 perf gate (`pytest tests/tools`) needs.
  #
  # Not covered here (host/SDK-bound, cannot be pinned by Nix):
  #   - the macOS system SDK (Accelerate / Metal / Foundation) — Xcode CLT;
  #   - the CUDA stack (Linux + NVIDIA driver) — enable via cudaPackages if wanted;
  #   - libDaisy / DaisySP — external board SDKs passed by -D path at configure.
  #
  # Usage:   nix develop          (then: make configure && make)
  #          nix develop -c bash  (run a one-off command in the env)
  # Commit flake.lock so the nixpkgs revision is pinned for everyone.
  description = "Spectral Engine — reproducible dev shell (desktop + embedded ARM perf toolchain)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          lib = pkgs.lib;

          # Python tooling — pins mirror tools/requirements.txt (numpy/xxhash/pytest/pyyaml).
          python = pkgs.python3.withPackages (ps: with ps; [
            numpy
            pyyaml
            pytest
            xxhash
          ]);

          # Always-on toolchain (desktop build + embedded perf model).
          common = with pkgs; [
            cmake
            gnumake
            ninja                 # optional faster generator (CMAKE_GENERATOR=Ninja)
            pkg-config
            libsndfile            # audio I/O
            llvmPackages.openmp   # libomp — OpenMP runtime for the CPU synth
            llvm                  # llvm-mca (M7 [modeled] cycles) + llvm-profdata (PGO)
            qemu                  # qemu-system-arm — M7 [measured] counts
            gcc-arm-embedded      # arm-none-eabi-gcc (+newlib); repo also vendors xpack 15.2.1
            ripgrep               # log-discipline build check
            dfu-util              # `make daisy-flash`
            git
            python
          ];

          # Host FFT: macOS uses Accelerate/vDSP (system SDK); Linux needs FFTW3.
          linuxOnly = with pkgs; [ fftw ];

          # macOS GPU/accel frameworks. Looked up defensively: on newer nixpkgs the
          # unified Apple SDK provides these through stdenv and may drop the explicit
          # `apple_sdk.frameworks.*` attrs, so a missing attr resolves to [] (no eval
          # break) rather than erroring.
          fw = pkgs.darwin.apple_sdk.frameworks or { };
          darwinFrameworks = lib.optionals pkgs.stdenv.isDarwin (
            builtins.filter (x: x != null)
              (map (n: fw.${n} or null) [ "Accelerate" "Metal" "Foundation" ])
          );
        in {
          default = pkgs.mkShell {
            packages = common
              ++ lib.optionals pkgs.stdenv.isLinux linuxOnly
              ++ darwinFrameworks;

            shellHook = ''
              echo "Spectral Engine dev shell"
              echo "  $(cmake --version | head -1)"
              echo "  $(qemu-system-arm --version | head -1)"
              echo "  llvm-mca: $(command -v llvm-mca || echo 'on PATH')"
              echo "Embedded cross-compiler: in-tree xpack (tools/toolchains/...) is used by"
              echo "the perf stack; a Nix arm-none-eabi-gcc is also on PATH as a fallback."
              echo "First time: git submodule update --init  (or: PYTHONPATH=tools \\"
              echo "  python -m spectral_tools.vendor submodule sync)"
            '';
          };
        });
    };
}
