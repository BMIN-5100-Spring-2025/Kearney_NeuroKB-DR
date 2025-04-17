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

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs_task_execution_role_v1"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "ecs_task_role_v1"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_role_policy" {
  name = "ecs_task_role_policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Effect   = "Allow"
        Resource = [
          "arn:aws:s3:::kearneyneurokb-dr",
          "arn:aws:s3:::kearneyneurokb-dr/*"
        ]
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "kearney_neurokb_dr_ecs_log_group" {
  name              = "/ecs/kearney_neurokb_dr"
  retention_in_days = 30  # Optional: Set log retention
}

resource "aws_ecs_task_definition" "kearney_neurokb_dr" {
  family                   = "kearney_neurokb_dr"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "kearney_neurokb_dr"
      image     = "061051226319.dkr.ecr.us-east-1.amazonaws.com/kearneyneurokb-dr:latest"
      essential = true
      environment = [
        {
          name  = "S3_BUCKET_ARN"
          value = "${aws_s3_bucket.Kearney_NeuroKB-DR.bucket}"
        },
        { name = "ENVIRONMENT"
          value = "FARGATE"
        },
        {
          "name": "RUN_ENV",
          "value": "fargate"
        },
        {
          "name": "S3_BUCKET_NAME",
          "value": "${aws_s3_bucket.Kearney_NeuroKB-DR.id}"
        },
        {
          "name":"DISEASE",
          "value":"1"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.kearney_neurokb_dr_ecs_log_group.name
          awslogs-region        = data.aws_region.current_region.name
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  ephemeral_storage {
    size_in_gib = 200
  }
}

resource "aws_s3_bucket_cors_configuration" "kearneyneurokb-dr_cors_configuration" {
  bucket = aws_s3_bucket.Kearney_NeuroKB-DR.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "POST", "PUT", "HEAD"]
    allowed_origins = ["http://localhost:3000", "bmin5100.com", "*.bmin5100"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}