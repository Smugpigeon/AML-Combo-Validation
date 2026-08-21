#!/usr/bin/env Rscript

# Run CopyKAT independently so a SCEVAN native failure cannot suppress its output.

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) stop(paste("missing", flag))
  args[[index + 1]]
}

input_dir <- value_after("--input-dir")
identity_input_dir <- value_after("--identity-input-dir")
output_dir <- normalizePath(value_after("--output-dir"), mustWork = FALSE)
upstream_script <- value_after("--upstream-script")
patient <- value_after("--patient")
ncores <- as.integer(value_after("--ncores"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
patient_work_dir <- file.path(output_dir, paste0(patient, "_work"))
dir.create(patient_work_dir, recursive = TRUE, showWarnings = FALSE)
setwd(patient_work_dir)
required_packages <- c(
  "dplyr", "Seurat", "copykat", "copykatRcpp", "Rcpp", "Rclusterpp",
  "parallel", "jsonlite"
)
suppressPackageStartupMessages(
  invisible(lapply(required_packages, library, character.only = TRUE))
)
source(upstream_script)

identity_inputs <- read.csv(
  file.path(identity_input_dir, paste0(patient, "_identity_inputs.csv")),
  stringsAsFactors = FALSE
)
known_normal_cells <- identity_inputs$cell_barcode[
  as.logical(identity_inputs$known_normal_reference)
]
if (length(known_normal_cells) < 20) {
  stop("fewer than 20 normal references")
}

object <- readRDS(file.path(input_dir, paste0(patient, ".RDS")))
object <- run_copyKat(
  object,
  known_normal_cells = known_normal_cells,
  plot = FALSE,
  ncores = ncores,
  genome = "hg20"
)
output <- data.frame(
  sample_id = patient,
  cell_barcode = rownames(object@meta.data),
  copyKat_output = as.character(object@meta.data$copyKat_output),
  stringsAsFactors = FALSE
)
write.csv(
  output,
  file.path(output_dir, paste0(patient, "_copykat_calls.csv")),
  row.names = FALSE,
  quote = TRUE
)
status <- list(
  patient_id = patient,
  status = "complete",
  n_cells = nrow(output),
  n_malignant = sum(output$copyKat_output == "malignant", na.rm = TRUE),
  n_healthy = sum(output$copyKat_output == "healthy", na.rm = TRUE),
  n_unresolved = sum(is.na(output$copyKat_output)),
  n_normal_references = length(known_normal_cells),
  ncores = ncores,
  genome = "hg20"
)
writeLines(
  jsonlite::toJSON(status, auto_unbox = TRUE, pretty = TRUE),
  file.path(output_dir, paste0(patient, "_copykat_status.json"))
)
writeLines(
  capture.output(sessionInfo()),
  file.path(output_dir, paste0(patient, "_R_SESSION_INFO.txt"))
)
