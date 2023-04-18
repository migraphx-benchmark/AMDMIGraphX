# Workflows

## `benchmark.yaml`

<p>
This workflow runs the MiGraphX performance benchmarks and generates comparison reports by comparing the results with the reference data.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [benchmarks.yml](https://github.com/migraphx-benchmark/actions/blob/main/.github/workflows/benchmarks.yml)

- ## Input Parameters
The workflow uses the following input parameters:

> - `rocm_version`: the version of ROCm to use for running the benchmarks.

> - `script_repo`: the repository containing the benchmark scripts.

> - `result_path`: the path where the benchmark results will be stored.

> - `result_repo`: the repository where the benchmark results will be pushed for comparison.

For more details, please refer to the [benchmark.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/benchmark.yaml) file in the repository.

---
## `history.yaml`

<p>
This workflow generates a report of the MiGraphX benchmark results between two dates and sends it to a specified email address. The report is also uploaded to a specified repository.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [history.yml](https://github.com/ROCmSoftwarePlatform/migraphx-benchmark/blob/main/.github/workflows/history.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `start_date`: Start date for results analysis.

> - `end_date`: End date for results analysis.

> - `history_repo`: Repository for history results between dates.

> - `benchmark_utils_repo`: Repository where benchmark utils are stored.

> - `organization`: Organization based on which location of files will be different.

For more details, please refer to the [history.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/history.yaml) file in the repository.

---
## `history_HTEC.yaml`

<p>
This workflow analyzes the benchmark results between two dates and generates a report of the performance comparison.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [history.yml](https://github.com/migraphx-benchmark/actions/blob/main/.github/workflows/history.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `start_date`: Start date for analysis.

> - `end_date`: End date for analysis.

> - `history_repo`: Repository for history results between dates.

> - `benchmark_utils_repo`: Repository where benchmark utils are stored.

> - `organization`: Organization based on which location of files will be different.

For more details, please refer to the [history_HTEC.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/history_HTEC.yaml) file in the repository.

---
## `miopen_database.yaml`

<p>
This workflow generates a MIOpen database and pushes it to a specified repository.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [miopen-db.yml](https://github.com/migraphx-benchmark/actions/blob/main/.github/workflows/miopen-db.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm release version.

> - `miopen_db_repo`: Repository for MIOpen database.

> - `script_repo`: Repository where script files are stored.

> - `saved_models_path`: Path to the saved models.

> - `test_results_dir`: Path to the test results.

- ## Environment Variables
The workflow uses the following environment variables:

> - `ROCM_VERSION`: ROCm version based on the release input.

> - `MIOPEN_PATH`: Path to the MIOpen databases.

> - `SCRIPT_PATH`: Path to the script files.

For more details, please refer to the [miopen_database.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/miopen_database.yaml) file in the repository.

---
## `performance.yaml`

<p>
This workflow runs performance tests on the MIGraphX repository and generates a report of the results.
</p>

- ## Trigger
The workflow will run reusable workflow [perf-test.yml](https://github.com/ROCmSoftwarePlatform/migraphx-benchmark/blob/main/.github/workflows/perf-test.yml) by the following events:

> - Pull requests opened, synchronized or closed on the `develop` branch.

> - Schedule: Runs every day of the week from Monday to Saturday at 6:00 AM.

> - Manual trigger through the "Run workflow" button in the Actions tab of the repository.

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm version to use for the performance tests.

> - `performance_reports_repo`: Repository where the performance reports are stored.

> - `benchmark_utils_repo`: Repository where the benchmark utilities are stored.

> - `organization`: Organization based on which location of files will be different.

> - `result_number`: Last N results.

> - `model_timeout`: If a model in the performance test script passes this threshold, it will be skipped.

> - `flags`: Command line arguments to be passed to the performance test script. Default is `-r`.

For more details, please refer to the [performance.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/performance.yaml) file in the repository.

---
## `performance_HTEC.yaml`

<p>
This workflow runs performance tests for the MIGraphX library and generates a report of the results.
</p>

- ## Trigger
The workflow will run reusable workflow [perf-test.yml](https://github.com/migraphx-benchmark/actions/blob/main/.github/workflows/perf-test.yml) by the following events:

> - A pull request being opened, synchronized or closed on the `develop` branch.

> - A manual trigger using the "Run workflow" button in the Actions tab of the repository.

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm version to use for the performance tests.

> - `performance_reports_repo`: Repository where the performance reports will be stored.

> - `benchmark_utils_repo`: Repository where benchmark utils are stored.

> - `organization`: Organization based on which location of files will be different.

> - `result_number`: The number of most recent results to use for comparison.

> - `model_timeout`: The maximum time in minutes to wait for each model before skipping it.

> - `flags`: Flags to use for the performance test script. Use -m for Max value, -s for Std dev, and -r for Threshold file.

For more details, please refer to the [performance_HTEC.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/performance_HTEC.yaml) file in the repository.

---
## `rocm-image-release.yaml`

<p>
This workflow builds a Docker image for the specified ROCm release version.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [rocm-release.yml](https://github.com/ROCmSoftwarePlatform/migraphx-benchmark/blob/main/.github/workflows/rocm-release.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm release version for building the Docker image.

> - `benchmark-utils_repo`: Repository for benchmark utils.

For more details, please refer to the [rocm-image-release.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/rocm-image-release.yaml) file in the repository.

---
## `rocm-image-release_HTEC.yaml`

<p>
This workflow builds a Docker image for a specified ROCm release version and pushes it to the specified repository. If image already exists nothing will happen, and there is also option to overwrite existing image.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [rocm-release.yml](https://github.com/migraphx-benchmark/actions/blob/main/.github/workflows/rocm-release.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm release version to build Docker image for.

> - `benchmark_utils_repo`: Repository where benchmark utils are stored.

> - `base_image`: Base image for ROCm Docker build.

> - `docker_image`: Docker image name for ROCm Docker build.

> - `build_navi`: Build number for the Navi architecture.

> - `organization`: The organization name used to determine the location of files.

> - `overwrite`: Specify whether to overwrite the Docker image if it already exists.

For more details, please refer to the [rocm-image-release_HTEC.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/rocm-image-release_HTEC.yaml) file in the repository.

---