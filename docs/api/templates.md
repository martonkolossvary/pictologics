# Templates API

The templates are YAML files of configurations in the package: `standard_configs.yaml` (the six standard configurations), `lv_configs.yaml` and `coronary_configs.yaml` (30 cardiac CT configurations each). `RadiomicsPipeline.from_template(name)` makes a pipeline from a template; these functions read the files themselves. See [The Template Files](../user_guide/configurations.md#the-template-files).

::: pictologics.templates
    options:
      members:
        - list_template_files
        - load_template_file
        - get_all_templates
        - get_standard_templates
        - get_template_metadata
