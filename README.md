# Standalone [Patatrack](https://patatrack.web.cern.ch/patatrack/wiki/) pixel tracking

## Table of contents

* [Introduction](#introduction)
* [Status](#status)
* [Quick recipe](#quick-recipe)
  * [Additional make targets](#additional-make-targets)
* [Code structure](#code-structure)
* [Build system](#build-system)
* [Contribution guide](#contribution-guide)

## Introduction

The purpose of this package is to explore various performance
portability solutions with the
[Patatrack](https://patatrack.web.cern.ch/patatrack/wiki/) pixel
tracking application. The version here corresponds to
[CMSSW_11_1_0_pre4_Patatrack](https://github.com/cms-patatrack/cmssw/tree/CMSSW_11_1_0_pre4_Patatrack).

The application is designed to require minimal dependencies on the system:
* GNU Make, `curl`, `md5sum`, `tar`
  * CMake for `kokkostest` and `kokkos` programs
* C++17 capable compiler that works with `nvcc`, in the current setup this pretty much means GCC 8
* CUDA 10.2 runtime and drivers (real drivers are not needed for building)

All other external dependencies (listed below) are downloaded and built automatically.
* [TBB](https://github.com/intel/tbb) (all programs)
* [CUB](https://nvlabs.github.io/cub/) (`cudatest` and `cuda` programs)
* [Eigen](http://eigen.tuxfamily.org/) (`cuda` program)
* [Kokkos](https://github.com/kokkos/kokkos) (`kokkostest` and `kokkos` programs)
* [Boost](https://www.boost.org/) (`alpakatest` and `alpaka` programs)
  * Boost libraries from the system can also be used, but they need to be newer than 1.65.1
* [Alpaka](https://github.com/alpaka-group/alpaka) (`alpakatest` and `alpaka` programs)

The input data set consists of a minimal binary dump of 1000 events of
ttbar+PU events from of
[/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DR-PUAvg50IdealConditions_IdealConditions_102X_upgrade2018_design_v9_ext1-v2/FEVTDEBUGHLT](http://opendata.cern.ch/record/12303)
dataset from the [CMS](https://cms.cern/)
[Open Data](http://opendata.cern.ch/docs/about-cms). The data are
downloaded automatically during the build process.

## Status

| Application  | Description                      | Framework          | Device framework   | Test code          | Raw2Cluster        | RecHit             | Pixel tracking     | Vertex             | Transfers to CPU   |
|--------------|----------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| `fwtest`     | Framework test                   | :heavy_check_mark: |                    | :heavy_check_mark: |                    |                    |                    |                    |                    |
| `cudatest`   | CUDA FW test                     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |                    |                    |
| `cuda`       | CUDA version (frozen)            | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `cudadev`    | CUDA version (development)       | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `cudauvm`    | CUDA version with managed memory | :heavy_check_mark: | :heavy_check_mark: |                    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `kokkostest` | Kokkos FW test                   | :heavy_check_mark: |                    | :heavy_check_mark: |                    |                    |                    |                    |                    |
| `kokkos`     | Kokkos version                   | :heavy_check_mark: |                    | :heavy_check_mark: | :white_check_mark: |                    |                    |                    |                    |
| `alpakatest` | Alpaka FW test                   | :heavy_check_mark: |                    | :white_check_mark: |                    |                    |                    |                    |                    |
| `alpaka`     | Alpaka version                   | :white_check_mark: |                    |                    | :white_check_mark: |                    |                    |                    |                    |

The "Device framework" refers to a mechanism similar to [`cms::cuda::Product`](src/cuda/CUDACore/Product.h) and [`cms::cuda::ScopedContext`](src/cuda/CUDACore/ScopedContext.h) to support chains of modules to use the same device and the same work queue.

## Quick recipe

```bash
# Build application with N-fold concurrency
$ make -j N cuda

# For CUDA installations elsewhere than /usr/local/cuda
$ make -j N cuda CUDA_BASE=/path/to/cuda

# Source environment
$ source env.sh

# Process 1000 events in 1 thread
$ ./cuda

# Command line arguments
$ ./cuda -h
./cuda: [--numberOfThreads NT] [--numberOfStreams NS] [--maxEvents ME] [--data PATH] [--transfer] [--validation] [--empty]

Options
 --numberOfThreads   Number of threads to use (default 1)
 --numberOfStreams   Number of concurrent events (default 0=numberOfThreads)
 --maxEvents         Number of events to process (default -1 for all events in the input file)
 --data              Path to the 'data' directory (default 'data' in the directory of the executable)
 --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)
 --validation        Run (rudimentary) validation at the end (implies --transfer)
 --empty             Ignore all producers (for testing only)
```

### Additional make targets

| Target                  | Description                                             |
|-------------------------|---------------------------------------------------------|
| `all` (default)         | Build all programs                                      |
| `format`                | Format the code with `clang-format`                     |
| `test`                  | Run all tests (each program with small number of events |
| `test_cpu`              | Run tests for CPU test programs                         |
| `test_cuda`             | Run tests for CUDA test programs                        |
| `clean`                 | Remove all build artifacts                              |
| `distclean`             | `clean` and remove all externals                        |
| `dataclean`             | Remove downloaded data files                            |
| `external_kokkos_clean` | Remove Kokkos build and installation directory          |

### Test program specific notes (if any)

#### `cuda`

This program is frozen to correspond to CMSSW_11_1_0_pre4.

#### `cudadev`

This program contains developments after CMSSW_11_1_0_pre4.

#### `cudauvm`

The purpose of this program is to test the performance of the CUDA managed memory.

#### `kokkos` and `kokkostest`

If `nvcc` is not in your `$PATH`, the build recipe is
```bash
$ make environment [CUDA_BASE=...]
$ source env.sh
$ make -j N kokkos [CUDA_BASE=...]
$ ./kokkos --cuda
```
* Note that if `CUDA_BASE` needs to be set, it needs to be set for both `make` commands.
* The CMake executable can be set with `CMAKE` in case the default one is too old.
* The backend(s) need to be set explicitly via command line parameters (`--serial` for CPU serial backend, `--cuda` for CUDA backend)
* Use of multiple threads (`--numberOfThreads`) has not been tested and likely does not work correctly. Concurrent events (`--numberOfStreams`) works.

## Code structure

The project is split into several programs, one (or more) for each
test case. Each test case has its own directory under [`src`](src)
directory. A test case contains the full application: framework, data
formats, device tooling, plugins for the algorithmic modules ran
by the framework, and the executable.

Each test program is structured as follows within `src/<program name>`
(examples point to [`cuda`](src/cuda)
* [`Makefile`](src/cuda/Makefile) that defines the actual build rules for the program
* [`Makefile.deps`](src/cuda/Makefile.deps) that declares the external dependencies of the program, and the dependencies between shared objects within the program
* [`plugins.txt`](src/cuda/plugins.txt) contains a simple mapping from module names to the plugin shared object names
  - In CMSSW such information is generated automatically by `scram`, in this project the original author was lazy to automate that
* [`bin/`](src/cuda/bin/) directory that contains all the framework code for the executable binary. These files should not need to be modified, except [`main.cc`](src/cuda/bin/main.cc) for changin the set of modules to run, and possibly more command line options
* `plugin-<PluginName>/` directories contain the source code for plugins. The `<PluginName>` part specifies the name of the plugin, and the resulting shared object file is `plugin<PluginName>.so`. Note that no other library or plugin may depend on a plugin (either at link time or even thourgh `#includ`ing a header). The plugins may only be loaded through the names of the modules by the [`PluginManager`](src/cuda/bin/PluginManager.h).
* `<LibraryName>/`: the remaining directories are for libraries. The `<LibraryName>` specifies the name of the library, and the resulting shared object file is `lib<LibraryName>.so`. Other libraries or plugins may depend on a library, in which case the dependence must be declared in [`Makefile.deps`](src/cuda/Makefile.deps).
  * [`CondFormats/`](src/cuda/CondFormats/):
  * [`CUDADataFormats/`](src/cuda/CUDADataFormats/): CUDA-specific data structures that can be passed from one module to another via the `edm::Event`. A given portability technology likely needs its own data format directory, the `CUDADataFormats` can be used as an example.
  * [`CUDACore/`](src/cuda/CUDACore/): Various tools for CUDA. A given portability technology likely needs its own tool directory, the `CUDACore` can be used as an example.
  * [`DataFormats/`](src/cuda/DataFormats/): mainly CPU-side data structures that can be passed from one module to another via the `edm::Event`. Some of these are produced by the [`edm::Source`](src/cuda/bin/Source.h) by reading the binary dumps. These files should not need to be modified. New classes may be added, but they should be independent of the portability technology.
  * [`Framework/`](src/cuda/Framework/): crude approximation of the CMSSW framework. Utilizes TBB tasks to orchestrate the processing of the events by the modules. These files should not need to be modified.
  * [`Geometry/`](src/cuda/Geometry/): geometry information, essentially handful of compile-time constants. May be modified.

For more detailed description of the application structure (mostly plugins) see
[CodeStructure.md](doc/CodeStructure.md)

## Build system

The build system is based on pure GNU Make. There are two levels of
Makefiles. The [top-level Makefile](Makefile) handles the building of
the entire project: it defines general build flags, paths to external
dependencies in the system, recipes to download and build the
externals, and targets for the test programs.

For more information see [BuildSystem.md](doc/BuildSystem.md).

## Contribution guide

Given that the approach of this project is to maintain many programs
in a single branch, in order to keep the commit history readable, each
commit should contain changes only for one test program, and the short
commit message should start with the program name, e.g. `[cuda]`. A
pull request may touch many test programs. General commits (e.g.
top-level Makefile or documentation) can be left without such a prefix.

When starting work for a new portability technology, the first steps
are to figure out the installation of the necessary external software
packages and the build rules (both can be adjusted later). It is
probably best to start by cloning the `fwtest` code for the new
program (e.g. `footest` for a technology `foo`), adjust the test
modules to exercise the API of the technology (see `cudatest` for
examples), and start crafting the tools package (`CUDACore` in
`cuda`). 

Pull requests are expected to build (`make all` succeeds) and pass
tests (`make test`). Programs to have build errors should primarily be
filtered out from `$(TARGETS)`, and failing tests should primarily be
removed from the set of tests run by default. Breakages can, however,
be accepted for short periods of time with a good justification.
