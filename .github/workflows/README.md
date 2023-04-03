# AMD MIGraphX

<p> AMD MIGraphX is AMD's graph inference engine that accelerates machine learning model inference. <br> AMD MIGraphX can be used by installing binaries directly or building from source code.
</p>

# Workflows

## `backup_test.yml`

<p>
This workflow backs up the performance benchmarks of the `migraphx-benchmark/performance-backup` repository <br> to a local directory, and pushes the changes to the repository.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `PERFORMANCE_PATH`: the path to the root directory.
- `PERFORMANCE_DIR`: the name of the directory where the benchmarks will be backed up.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Checkout the `migraphx-benchmark/performance-backup` repository to the local `PERFORMANCE_DIR` directory.
2. Copy the performance benchmarks to the local directory.
3. Push the changes to the `migraphx-benchmark/performance-backup` repository.
```

For more details, please refer to the [backup_test.yml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/backup_test.yml) file in the repository.

---

## `backup_test_pr.yaml`

<p>
This workflow triggers the `backup.yml` workflow in the `migraphx-benchmark/actions` repository <br> whenever a pull request is closed on the `develop` branch.
</p>

- ### Trigger

```
The workflow is triggered when a pull request is closed on the `develop` branch.
```

- ### Secrets


The workflow uses the following secrets:

```
- `gh_token`: the GitHub token used to authenticate and authorize access to the repository.
```

- ### Jobs

The following job is executed in the workflow:
```
1. Uses the `backup.yml` workflow from the `migraphx-benchmark/actions` repository.
2. Authenticates using the `gh_token` secret.
```

For more details, please refer to the [backup_test_pr.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/backup_test_pr.yaml) file in the repository.

---

## `benchmark.yaml`

<p>
This workflow runs the MiGraphX performance benchmarks and generates comparison reports by comparing the results <br> with the reference data.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow uses the following input parameters:

```
- `rocm_version`: the version of ROCm to use for running the benchmarks.
- `script_repo`: the repository containing the benchmark scripts.
- `result_path`: the path where the benchmark results will be stored.
- `result_repo`: the repository where the benchmark results will be pushed for comparison.
```
For more details, please refer to the [benchmark.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/benchmark.yaml) file in the repository.

---
## `history.yml`

<p>
This workflow generates a report of the MiGraphX benchmark results between two dates and sends it to <br> a specified email address. The report is also uploaded to a specified repository.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository. 
```
- ### Input Parameters


The workflow requires the following inputs:
```
- `start_date`: Start date for results analysis.
- `end_date`: End date for results analysis.
- `history_repo`: Repository for history results between dates.
- `benchmark_utils_repo`: Repository where benchmark utils are stored.
- `organization`: Organization based on which location of files will be different.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `MAIL_USERNAME`: the username for the email account used to send the report.
- `MAIL_PASSWORD`: the password for the email account used to send the report.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Clone the `migraphx-reports` repository.
4. Generate the report.
5. Send the report to the specified email address.
6. Push the report to the specified repository.
```

For more details, please refer to the [history.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/history.yaml) file in the repository.

---
## `history_HTEC.yaml`

<p>
This workflow analyzes the benchmark results between two dates and generates a report of the performance comparison. <br> It uses the `migraphx-benchmark/actions/.github/workflows/history.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `start_date`: Start date for analysis.
- `end_date`: End date for analysis.
- `history_repo`: Repository for history results between dates.
- `benchmark_utils_repo`: Repository where benchmark utils are stored.
- `organization`: Organization based on which location of files will be different.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `MAIL_USERNAME`: the username for the email account used to send the report.
- `MAIL_PASSWORD`: the password for the email account used to send the report.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Clone the `migraphx-reports` repository.
4. Generate the report.
5. Send the report to the specified email address.
6. Push the report to the specified repository.
```

For more details, please refer to the [history.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/history.yaml) file in the repository.

---
## `miopen_database.yaml`

<p>
This workflow generates a MIOpen database and pushes it to a specified repository. It uses the <br> `migraphx-benchmark/actions/.github/workflows/miopen-db.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `rocm_release`: ROCm release version.
- `miopen_db_repo`: Repository for MIOpen database.
- `script_repo`: Repository where script files are stored.
- `saved_models_path`: Path to the saved models.
- `test_results_dir`: Path to the test results.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `ROCM_VERSION`: ROCm version based on the release input.
- `MIOPEN_PATH`: Path to the MIOpen databases.
- `SCRIPT_PATH`: Path to the script files.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `rocm-migraphx` repository.
3. Clone the `miopen-databases` repository.
4. Generate the MIOpen database.
5. Push the MIOpen database to the specified repository.
```

For more details, please refer to the [miopen_database.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/miopen_database.yaml) file in the repository.

---
## `performance.yaml`

<p>
This workflow runs performance tests on the MIGraphX repository and generates a report of the results. <br> It uses the `migraphx-benchmark/actions/.github/workflows/perf-test.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered by the following events:
- Pull requests opened, synchronized or closed on the `develop` branch.
- Schedule: Runs every day of the week from Monday to Saturday at 6:00 AM.
- Manual trigger through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `rocm_release`: ROCm version to use for the performance tests.
- `performance_reports_repo`: Repository where the performance reports are stored.
- `benchmark_utils_repo`: Repository where the benchmark utilities are stored.
- `organization`: Organization based on which location of files will be different.
- `result_number`: Last N results.
- `model_timeout`: If a model in the performance test script passes this threshold, it will be skipped.
- `flags`: Command line arguments to be passed to the performance test script. Default is `-r`.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `gh_token`: Personal access token used for authentication with GitHub.
- `mail_user`: The username for the email account used to send the report.
- `mail_pass`: The password for the email account used to send the report.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Clone the `migraphx-reports` repository.
4. Run the performance tests.
5. Generate the report.
6. Send the report to the specified email address.
7. Push the report to the specified repository.
```

For more details, please refer to the [performance.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/performance.yaml) file in the repository.

---
## `performance_HTEC.yaml`

<p>
This workflow runs performance tests for the MIGraphX library and generates a report of the results. <br> It uses the `migraphx-benchmark/actions/.github/workflows/perf-test.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered by the following events:

- A pull request being opened, synchronized or closed on the `develop` branch.
- A manual trigger using the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `rocm_release`: ROCm version to use for the performance tests.
- `performance_reports_repo`: Repository where the performance reports will be stored.
- `benchmark_utils_repo`: Repository where benchmark utils are stored.
- `organization`: Organization based on which location of files will be different.
- `result_number`: The number of most recent results to use for comparison.
- `model_timeout`: The maximum time in minutes to wait for each model before skipping it.
- `flags`: Flags to use for the performance test script. Use -m for Max value, -s for Std dev, and -r for Threshold file.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `gh_token`: GitHub token used to push the report to the specified repository.
- `mail_user`: The username for the email account used to send the report.
- `mail_pass`: The password for the email account used to send the report.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Clone the `migraphx-reports` repository.
4. Run the performance tests.
5. Generate the report.
6. Send the report to the specified email address.
7. Push the report to the specified repository.
```

For more details, please refer to the [performance_HTEC.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/performance_HTEC.yaml) file in the repository.

---
## `rocm-image-release.yaml`

<p>
This workflow builds a Docker image for the specified ROCm release version using the <br> `migraphx-benchmark/.github/workflows/rocm-release.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `rocm_release`: ROCm release version for building the Docker image.
- `benchmark-utils_repo`: Repository for benchmark utils.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `gh_token`: the access token for the GitHub bot account used to run the workflow.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Build the Docker image for the specified ROCm release version.
4. Push the Docker image to Docker Hub.
```

For more details, please refer to the [rocm-image-release.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/rocm-image-release.yaml) file in the repository.

---
## `rocm-image-release_HTEC.yaml`

<p>
This workflow builds a Docker image for a specified ROCm release version and pushes it to the specified repository.<br> It uses the `migraphx-benchmark/actions/.github/workflows/rocm-release.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `rocm_release`: ROCm release version to build Docker image for.
- `benchmark_utils_repo`: Repository where benchmark utils are stored.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `gh_token`: GitHub token used to push the Docker image to the repository.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Build the Docker image for the specified ROCm release version.
4. Push the Docker image to the specified repository.
```

For more details, please refer to the [rocm-image-release_HTEC.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/rocm-image-release_HTEC.yaml) file in the repository.

---

## `rocm-image-release_HTEC_dj.yaml`

<p>
This workflow builds a Docker image for the ROCm release and pushes it to a container registry. <br> It uses the `migraphx-benchmark/actions/.github/workflows/rocm-release-htec-dj.yml` workflow.
</p>

- ### Trigger

```
The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository.
```

- ### Input Parameters


The workflow requires the following inputs:

```
- `rocm_release`: The version of the ROCm release.
- `benchmark_utils_repo`: The repository where benchmark utils are stored.
- `base_image`: The base image for the ROCm Docker build.
- `docker_image`: The name of the Docker image for the ROCm Docker build.
- `build_navi`: The build number for the Docker image.
- `organization`: The organization based on which the location of files will be different.
- `overwrite`: A boolean value that specifies whether to overwrite the Docker image if it already exists.
```

- ### Environment Variables


The workflow uses the following environment variables:

```
- `gh_token`: The personal access token used to push the Docker image to the container registry.
```

- ### Steps

The following steps are executed in the workflow:
```
1. Set up the environment.
2. Clone the `migraphx-benchmark-utils` repository.
3. Build the Docker image using the specified ROCm release version and base image.
4. Tag the Docker image with the specified build number and organization.
5. Push the Docker image to the container registry.
6. (Optional) Remove the Docker image from the local machine.
```

For more details, please refer to the [rocm-image-release_HTEC_dj.yaml](https://github.com/migraphx-benchmark/AMDMIGraphX/blob/develop/.github/workflows/rocm-image-release_HTEC_dj.yaml) file in the repository.

---

## Using the Workflows 
<p>
To use these workflows, simply push changes to the repository and the appropriate workflow will be triggered automatically.<br> You can also trigger the workflows manually by clicking the "Run workflow" button in the Actions tab.

If you need to modify the workflows, you can do so by editing the `.github/workflows/*.yml` files in the repository. <br> Be sure to test your changes before committing them.

Some of the workflows require environment variables or secrets to be set in order to function correctly. <br> You can set these values in the repository or organization settings under "Secrets".
</p>