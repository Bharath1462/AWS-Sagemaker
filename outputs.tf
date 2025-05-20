output "s3_bucket_name" {
  value = module.s3_bucket.bucket_name
}

output "sagemaker_role_arn" {
  value = module.iam_role.role_arn
}
