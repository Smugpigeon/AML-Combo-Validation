#!/usr/bin/env Rscript

# Reproduce the published scTherapy ScType/CopyKAT/SCEVAN identity ensemble.
# This wrapper calls the upstream implementation; it does not reimplement or
# vendor upstream GPL code.

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) stop(paste("missing", flag))
  args[[index + 1]]
}

input_dir <- value_after("--input-dir")
output_dir <- value_after("--output-dir")
upstream_script <- value_after("--upstream-script")
custom_marker <- value_after("--custom-marker")
patients <- strsplit(value_after("--patients"), ",", fixed = TRUE)[[1]]
ncores <- as.integer(value_after("--ncores"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
required_packages <- c(
  "dplyr", "Seurat", "HGNChelper", "openxlsx", "copykat", "copykatRcpp",
  "ggplot2", "SCEVAN", "yaGST", "cowplot", "Rcpp", "Rclusterpp",
  "parallel", "biomaRt", "logger", "httr", "jsonlite", "readr", "future"
)
suppressPackageStartupMessages(
  invisible(lapply(required_packages, library, character.only = TRUE))
)
source(upstream_script)

normalize_call <- function(values) {
  values <- tolower(trimws(as.character(values)))
  ifelse(values %in% c("malignant", "aneuploid"), "malignant",
         ifelse(values %in% c("healthy", "normal", "diploid"), "healthy", NA))
}

majority_vote <- function(sctype, copykat, scevan) {
  calls <- c(sctype, copykat, scevan)
  calls <- calls[!is.na(calls)]
  if (length(calls) < 2) return(NA_character_)
  counts <- table(calls)
  if (max(counts) < 2) return(NA_character_)
  names(counts)[which.max(counts)]
}

has_scaled_rna <- function(object) {
  assay <- object[["RNA"]]
  if (inherits(assay, "Assay5")) {
    return("scale.data" %in% Layers(assay) && nrow(LayerData(assay, layer = "scale.data")) > 0)
  }
  nrow(GetAssayData(object, assay = "RNA", slot = "scale.data")) > 0
}

statuses <- list()
for (patient in patients) {
  input_path <- file.path(input_dir, paste0(patient, ".RDS"))
  output_csv <- file.path(output_dir, paste0(patient, "_identity_calls.csv"))
  output_rds <- file.path(output_dir, paste0(patient, "_identity_ensemble.RDS"))
  message("[identity] ", patient)
  status <- tryCatch({
    object <- readRDS(input_path)
    if (!("seurat_clusters" %in% colnames(object@meta.data))) {
      if (!("RNA_snn_res.0.5" %in% colnames(object@meta.data))) {
        stop("no cluster column in Seurat metadata")
      }
      object$seurat_clusters <- factor(object$RNA_snn_res.0.5)
    }
    Idents(object) <- "seurat_clusters"
    if (!has_scaled_rna(object)) {
      object <- ScaleData(object, features = rownames(object), verbose = FALSE)
    }

    object <- run_sctype(
      object,
      known_tissue_type = "Immune system",
      plot = FALSE,
      name = "sctype_classification"
    )
    general_labels <- as.character(object@meta.data$sctype_classification)
    t_cell <- grepl("(^|[^A-Za-z])T[- ]?cells?", general_labels, ignore.case = TRUE)
    known_normal_cells <- rownames(object@meta.data)[t_cell]
    if (length(known_normal_cells) < 20) {
      stop(paste("fewer than 20 ScType T-cell normal references:", length(known_normal_cells)))
    }

    object <- run_sctype(
      object,
      known_tissue_type = "AML",
      custom_marker_file = custom_marker,
      plot = FALSE,
      name = "sctype_malignant_healthy"
    )
    object <- run_SCEVAN(
      object,
      known_normal_cells = known_normal_cells,
      plot = FALSE,
      ncores = ncores,
      all_pred = FALSE,
      organism_ = "human"
    )
    object <- run_copyKat(
      object,
      known_normal_cells = known_normal_cells,
      plot = FALSE,
      ncores = ncores,
      genome = "hg20"
    )

    meta <- object@meta.data
    sctype_call <- normalize_call(meta$sctype_malignant_healthy)
    copykat_call <- normalize_call(meta$copyKat_output)
    scevan_call <- normalize_call(meta$SCEVAN_output)
    ensemble <- mapply(majority_vote, sctype_call, copykat_call, scevan_call)
    meta$ensemble_output <- ensemble
    meta$known_normal_reference <- rownames(meta) %in% known_normal_cells
    object@meta.data <- meta

    output <- data.frame(
      sample_id = patient,
      cell_barcode = rownames(meta),
      seurat_cluster = as.character(meta$seurat_clusters),
      sctype_classification = as.character(meta$sctype_classification),
      known_normal_reference = meta$known_normal_reference,
      sctype_malignant_healthy = sctype_call,
      copyKat_output = copykat_call,
      SCEVAN_output = scevan_call,
      ensemble_output = ensemble,
      stringsAsFactors = FALSE
    )
    write.csv(output, output_csv, row.names = FALSE, quote = TRUE)
    saveRDS(object, output_rds)
    list(patient_id = patient, status = "complete", n_cells = nrow(output),
         n_normal_references = length(known_normal_cells))
  }, error = function(error) {
    message("[identity:error] ", patient, ": ", conditionMessage(error))
    list(patient_id = patient, status = "failed", error = conditionMessage(error))
  })
  statuses[[patient]] <- status
  writeLines(jsonlite::toJSON(statuses, auto_unbox = TRUE, pretty = TRUE),
             file.path(output_dir, "run_status.json"))
}

writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_SESSION_INFO.txt"))
if (any(vapply(statuses, function(item) item$status != "complete", logical(1)))) {
  quit(status = 2)
}
