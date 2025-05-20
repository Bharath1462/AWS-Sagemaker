provider "aws" {
  region = var.aws_region
}

module "s3_bucket" {
  source = "./modules/s3_bucket"
  bucket_prefix = "llm-data"
}

module "iam_role" {
  source = "./modules/iam_role"
}

module "sagemaker" {
  source            = "./modules/sagemaker"
  sagemaker_role_arn = module.iam_role.role_arn
}
