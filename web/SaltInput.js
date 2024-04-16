import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "SaltInput",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function(
      name,
      value,
      old_value,
      widget,
    ) {
      onWidgetChanged?.apply(this, arguments);
      if (name == "input_type" && ["IMAGE", "FILE"].indexOf(value) >= 0) {
        this.widgets[3].value = api.inputs[0];
        this.widgets[3].options.values = api.inputs;
        this.widgets[3].type = "combo";
      } else if (name == "input_type") {
        if (api.inputs.indexOf(this.widgets[0].value) >= 0) {
          this.widgets[3].value = "";
        }
        // If the old value was a file, clear the values
        this.widgets[3].type = "text";
        this.widgets[3].options.values = undefined;
      }
      console.log(name, value, old_value, widget);
    };
  },
});
