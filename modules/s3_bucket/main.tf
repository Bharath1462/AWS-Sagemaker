resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "llm_data" {
  bucket = "${var.bucket_prefix}-${random_id.suffix.hex}"
}

variable "bucket_prefix" {
  type = string
}
