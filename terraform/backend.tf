terraform {
  backend "s3" {
    bucket        = "bmin5100-terraform-state"
    key           = "Sophie.Kearney@Pennmedicine.upenn.edu-Kearney_NeuroKB-DR/terraform.tfstate"
    region        = "us-east-1"
    encrypt       = true
  }
}