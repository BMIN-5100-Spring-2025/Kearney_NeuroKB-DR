resource "aws_s3_bucket" "Kearney_NeuroKB-DR" {
  bucket = "kearneyneurokb-dr"

  tags = {
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

resource "aws_s3_bucket_ownership_controls" "Kearney_NeuroKB-DR_ownership_controls" {
  bucket = aws_s3_bucket.Kearney_NeuroKB-DR.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "Kearney_NeuroKB-DR_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.Kearney_NeuroKB-DR_ownership_controls]

  bucket = aws_s3_bucket.Kearney_NeuroKB-DR.id
  acl    = "private"
}

resource "aws_s3_bucket_lifecycle_configuration" "Kearney_NeuroKB-DR_expiration" {
  bucket = aws_s3_bucket.Kearney_NeuroKB-DR.id

  rule {
    id      = "compliance-retention-policy"
    status  = "Enabled"

    expiration {
	  days = 100
    }
  }
}

resource "aws_ecr_repository" "Kearney_NeuroKB-DR" {
  name                 = "kearneyneurokb-dr"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
