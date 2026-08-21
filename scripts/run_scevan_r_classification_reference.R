#!/usr/bin/env Rscript

# Produce a true SCEVAN classification reference from the pinned R implementation.
# Run one patient per process because a native VegaMC crash cannot be caught by R.

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) stop(paste("missing", flag))
  args[[index + 1]]
}

input_dir <- value_after("--input-dir")
identity_input_dir <- value_after("--identity-input-dir")
output_dir <- normalizePath(value_after("--output-dir"), mustWork = FALSE)
scevan_script <- value_after("--scevan-script")
patient <- value_after("--patient")
ncores <- as.integer(value_after("--ncores"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
patient_work_dir <- file.path(output_dir, paste0(patient, "_work"))
dir.create(patient_work_dir, recursive = TRUE, showWarnings = FALSE)
setwd(patient_work_dir)
options(
  error = function() {
    traceback(100)
    dump.frames("last.dump", to.file = TRUE)
    quit(save = "no", status = 1, runLast = FALSE)
  }
)

required_packages <- c(
  "Seurat", "SCEVAN", "yaGST", "cowplot", "Rcpp", "Rclusterpp",
  "parallel", "biomaRt", "jsonlite"
)
suppressPackageStartupMessages(
  invisible(lapply(required_packages, library, character.only = TRUE))
)
source(scevan_script)

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
count_mtx <- object@assays$RNA$counts
results <- pipelineCNA1(
  count_mtx,
  sample = patient,
  par_cores = ncores,
  SUBCLONES = FALSE,
  plotTree = FALSE,
  norm_cell = known_normal_cells,
  organism = "human"
)

scevan_output <- ifelse(
  results$class == "tumor",
  "malignant",
  ifelse(results$class == "normal", "healthy", NA_character_)
)
output <- data.frame(
  sample_id = patient,
  cell_barcode = rownames(results),
  scevan_r_class = as.character(results$class),
  SCEVAN_output = scevan_output,
  stringsAsFactors = FALSE
)
write.csv(
  output,
  file.path(output_dir, paste0(patient, "_scevan_r_classification.csv")),
  row.names = FALSE,
  quote = TRUE
)
status <- list(
  patient_id = patient,
  status = "complete",
  n_cells = nrow(output),
  n_evaluable = sum(!is.na(output$SCEVAN_output)),
  n_malignant = sum(output$SCEVAN_output == "malignant", na.rm = TRUE),
  n_healthy = sum(output$SCEVAN_output == "healthy", na.rm = TRUE),
  n_normal_references = length(known_normal_cells),
  ncores = ncores,
  segmentation_class = TRUE,
  beta_vega = 0.5
)
writeLines(
  jsonlite::toJSON(status, auto_unbox = TRUE, pretty = TRUE),
  file.path(output_dir, paste0(patient, "_scevan_r_status.json"))
)
writeLines(
  capture.output(sessionInfo()),
  file.path(output_dir, paste0(patient, "_R_SESSION_INFO.txt"))
)
