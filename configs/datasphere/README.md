# DataSphere manifests

Запускать из корня репозитория через `scripts/datasphere_runner.py -c <path>`.
Пути внутри YAML относятся к `/job`, не к каталогу manifest.

| Каталог | Число YAML | Назначение |
|---|---:|---|
| `cv/` | 22 | Direct temporal CV variants |
| `submissions/` | 11 | Final-training/submission recipes разных поколений |
| `smoke/` | 1 | Existing cloud smoke |
| `archive/` | 56 | SSL/hurdle/reference и ранние research jobs |

Название manifest не подтверждает Public score. См.
[submission provenance](../../docs/SUBMISSION_PROVENANCE.md).
Исторические missing inputs и transfer-layout ограничения:
[cleanup checks](../../docs/CLEANUP_CHECKS.md).
Точный запуск/мониторинг: [runbook](../../DATASPHERE_AGENT_RUNBOOK.md).
