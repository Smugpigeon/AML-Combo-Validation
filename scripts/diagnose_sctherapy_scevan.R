#!/usr/bin/env Rscript

# Diagnose the pinned scTherapy SCEVAN path without changing its thresholds.
# The script preserves preprocessing dimensions, the raw VegaMC output and an
# R traceback so an upstream implementation failure can be repaired audibly.

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) stop(paste("missing", flag))
  args[[index + 1]]
}

input_dir <- value_after("--input-dir")
output_dir <- normalizePath(
  value_after("--output-dir"),
  mustWork = FALSE
)
upstream_script <- value_after("--upstream-script")
custom_marker <- value_after("--custom-marker")
sctype_dir <- value_after("--sctype-dir")
scevan_script <- value_after("--scevan-script")
patient <- value_after("--patient")
ncores <- as.integer(value_after("--ncores"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)
options(
  error = function() {
    cat("\n--- R traceback ---\n", file = stderr())
    traceback(100)
    dump.frames("last.dump", to.file = TRUE)
    quit(save = "no", status = 1, runLast = FALSE)
  },
  show.error.calls = TRUE
)

required_packages <- c(
  "dplyr", "Seurat", "HGNChelper", "openxlsx", "copykat", "copykatRcpp",
  "ggplot2", "SCEVAN", "yaGST", "cowplot", "Rcpp", "Rclusterpp",
  "parallel", "biomaRt", "logger", "httr", "jsonlite", "readr", "future"
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
source(scevan_script)

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

write_json <- function(value, name) {
  writeLines(
    jsonlite::toJSON(value, auto_unbox = TRUE, pretty = TRUE, na = "null"),
    file.path(output_dir, name)
  )
}

input_path <- file.path(input_dir, paste0(patient, ".RDS"))
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
t_cell <- grepl(
  "T[- _]?cells?|NKT|CD4|CD8",
  general_labels,
  ignore.case = TRUE,
  perl = TRUE
)
known_normal_cells <- rownames(object@meta.data)[t_cell]
if (length(known_normal_cells) < 20) {
  stop(paste(
    "fewer than 20 ScType T-cell normal references:",
    length(known_normal_cells)
  ))
}
object <- run_sctype(
  object,
  known_tissue_type = "AML",
  custom_marker_file = custom_marker,
  plot = FALSE,
  name = "sctype_malignant_healthy"
)

count_mtx <- object@assays$RNA$counts
res_proc <- SCEVAN:::preprocessingMtx(
  count_mtx,
  patient,
  par_cores = ncores,
  findConfident = FALSE,
  AdditionalGeneSets = NULL,
  SCEVANsignatures = TRUE,
  organism = "human"
)
count_mtx_norm <- res_proc$count_mtx_norm
annot_mtx <- res_proc$count_mtx_annot
write_json(
  list(
    patient_id = patient,
    raw_genes = nrow(count_mtx),
    raw_cells = ncol(count_mtx),
    normalized_genes = nrow(count_mtx_norm),
    normalized_cells = ncol(count_mtx_norm),
    annotation_rows = nrow(annot_mtx),
    annotation_columns = ncol(annot_mtx),
    duplicated_normalized_gene_names = sum(duplicated(rownames(count_mtx_norm))),
    duplicated_annotation_gene_names = sum(duplicated(annot_mtx[, 4])),
    known_normal_references = length(known_normal_cells),
    known_normal_references_after_filter = sum(
      colnames(count_mtx_norm) %in% known_normal_cells
    )
  ),
  "preprocessing_diagnostics.json"
)

# Keep only the matrices required by classifyTumorCells1. The diagnostic path
# otherwise retains the full Seurat object and a duplicate preprocessing list,
# which can exceed the memory footprint of the original upstream call.
rm(object, count_mtx, res_proc, general_labels, t_cell)
invisible(gc(verbose = FALSE))

original_get_breaks <- SCEVAN:::getBreaksVegaMC
getBreaksVegaMC <- function(mtx, chr_vect, sample = "", beta_vega = 0.5) {
  write_json(
    list(
      matrix_rows = nrow(mtx),
      matrix_columns = ncol(mtx),
      chromosome_vector_length = length(chr_vect),
      chromosome_levels = length(unique(chr_vect)),
      missing_chromosomes = sum(is.na(chr_vect)),
      beta = beta_vega
    ),
    "vega_input_diagnostics.json"
  )
  tryCatch(
    original_get_breaks(mtx, chr_vect, sample, beta_vega),
    error = function(error) {
      vega_path <- paste0("./output/ ", sample, " vega_output")
      if (file.exists(vega_path)) {
        file.copy(
          vega_path,
          file.path(output_dir, "vega_output_raw.tsv"),
          overwrite = TRUE
        )
        lines <- readLines(vega_path, warn = FALSE)
        fields <- lengths(strsplit(lines, "\t", fixed = TRUE))
        write.csv(
          data.frame(
            line_number = seq_along(lines),
            field_count = fields,
            line = lines,
            stringsAsFactors = FALSE
          ),
          file.path(output_dir, "vega_output_line_audit.csv"),
          row.names = FALSE,
          quote = TRUE
        )
      }
      stop(error)
    }
  )
}

result <- classifyTumorCells1(
  count_mtx_norm,
  annot_mtx,
  patient,
  par_cores = ncores,
  ground_truth = NULL,
  norm_cell_names = known_normal_cells,
  SEGMENTATION_CLASS = TRUE,
  SMOOTH = TRUE,
  beta_vega = 0.5
)
write_json(
  list(
    patient_id = patient,
    status = "complete",
    tumor_cells = length(result$tum_cells),
    confident_normal_cells = length(result$confidentNormal)
  ),
  "diagnostic_result.json"
)
