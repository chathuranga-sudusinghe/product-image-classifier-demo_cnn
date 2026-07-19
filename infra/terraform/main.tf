terraform {
  required_version = ">= 1.5.0"
}

# ---------------------------------------------------------
# Placeholder infrastructure file for the mini project
# ---------------------------------------------------------
# This project does not provision real cloud resources yet.
# The file exists to keep a clean infrastructure structure
# and to show where future Terraform code can be added.

locals {
  project_name = "product-image-classifier-demo"
  environment  = "dev"
}

output "project_name" {
  description = "Project name for this mini project."
  value       = local.project_name
}

output "environment" {
  description = "Environment name for this mini project."
  value       = local.environment
}