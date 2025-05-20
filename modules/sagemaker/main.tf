variable "sagemaker_role_arn" {
  type = string
}

output "info" {
  value = "Use this role ARN in Python SDK to run training/deployment"
}
