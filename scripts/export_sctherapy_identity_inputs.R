#!/usr/bin/env Rscript

# Export ScType calls and confident-normal barcodes without running CNA tools.

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) stop(paste("missing", flag))
  args[[index + 1]]
}

input_dir <- value_after("--input-dir")
output_dir <- normalizePath(value_after("--output-dir"), mustWork = FALSE)
upstream_script <- value_after("--upstream-script")
custom_marker <- value_after("--custom-marker")
sctype_dir <- value_after("--sctype-dir")
patients <- strsplit(value_after("--patients"), ",", fixed = TRUE)[[1]]

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
required_packages <- c(
  "dplyr", "Seurat", "HGNChelper", "openxlsx", "jsonlite", "readr"
)
suppressPackageStartupMessages(
  invisible(lapply(required_packages, library, character.only = TRUE))
)
source(upstream_script)
sctype_source <- function() {
  source(file.path(sctype_dir, "auto_detect_tissue_type.R"))
  source(file.path(sctype_dir, "gene_sets_prepare.R"))
  source(file.path(sctype_dir, "sctype_score_.R"))
  file.path(sctype_dir, "ScTypeDB_full.xlsx")
}

has_scaled_rna <- function(object) {
  assay <- object[["RNA"]]
  if (inherits(assay, "Assay5")) {
    return(
      "scale.data" %in% Layers(assay) &&
        nrow(LayerData(assay, layer = "scale.data")) > 0
    )
  }
  nrow(GetAssayData(object, assay = "RNA", slot = "scale.data")) > 0
}

statuses <- list()
for (patient in patients) {
  message("[identity-input] ", patient)
  status <- tryCatch({
    object <- readRDS(file.path(input_dir, paste0(patient, ".RDS")))
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
    known_normal <- grepl(
      "T[- _]?cells?|NKT|CD4|CD8",
      general_labels,
      ignore.case = TRUE,
      perl = TRUE
    )
    if (sum(known_normal) < 20) {
      stop(paste("fewer than 20 ScType T-cell normal references:", sum(known_normal)))
    }
    object <- run_sctype(
      object,
      known_tissue_type = "AML",
      custom_marker_file = custom_marker,
      plot = FALSE,
      name = "sctype_malignant_healthy"
    )

    output <- data.frame(
      sample_id = patient,
      cell_barcode = rownames(object@meta.data),
      seurat_cluster = as.character(object@meta.data$seurat_clusters),
      sctype_classification = general_labels,
      known_normal_reference = known_normal,
      sctype_malignant_healthy = as.character(
        object@meta.data$sctype_malignant_healthy
      ),
      stringsAsFactors = FALSE
    )
    write.csv(
      output,
      file.path(output_dir, paste0(patient, "_identity_inputs.csv")),
      row.names = FALSE,
      quote = TRUE
    )
    writeLines(
      output$cell_barcode[output$known_normal_reference],
      file.path(output_dir, paste0(patient, "_normal_cells.txt"))
    )
    list(
      patient_id = patient,
      status = "complete",
      n_cells = nrow(output),
      n_normal_references = sum(output$known_normal_reference)
    )
  }, error = function(error) {
    message("[identity-input:error] ", patient, ": ", conditionMessage(error))
    list(patient_id = patient, status = "failed", error = conditionMessage(error))
  })
  statuses[[patient]] <- status
  writeLines(
    jsonlite::toJSON(statuses, auto_unbox = TRUE, pretty = TRUE),
    file.path(output_dir, "run_status.json")
  )
  if (exists("object", inherits = FALSE)) rm(object)
  invisible(gc(verbose = FALSE))
}

writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_SESSION_INFO.txt"))
if (any(vapply(statuses, function(item) item$status != "complete", logical(1)))) {
  quit(status = 2)
}
