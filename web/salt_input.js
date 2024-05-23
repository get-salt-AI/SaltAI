import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
};

app.registerExtension({
    name: "SaltAI.SaltInput",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData?.name !== "SaltInput") {
            return;
        }
        nodeType.prototype.onNodeCreated = function() {
            // widgetName and type are not provided

            const widgetName = "input_value";
            const type = "file";

            const pathWidget = this.widgets.find((w) => w.name === widgetName);
            const fileInput = document.createElement("input");

            const oldRemoved = this.onRemoved;
            this.onRemoved = () => {
                oldRemoved.apply(this);
                fileInput?.remove();
            };

            if (type == "folder") {
                Object.assign(fileInput, {
                    type: "file",
                    style: "display: none",
                    webkitdirectory: true,
                    onchange: async () => {
                        const directory = fileInput.files[0].webkitRelativePath;
                        const i = directory.lastIndexOf('/');
                        if (i <= 0) {
                            throw "No directory found";
                        }
                        const path = directory.slice(0,directory.lastIndexOf('/'))
                        if (pathWidget?.options.values.includes(path)) {
                            alert("A folder of the same name already exists");
                            return;
                        }
                        let successes = 0;
                        for(const file of fileInput.files) {
                            if (await uploadFile(file).status == 200) {
                                successes++;
                            } else {
                                //Upload failed, but some prior uploads may have succeeded
                                //Stop future uploads to prevent cascading failures
                                //and only add to list if an upload has succeeded
                                if (successes > 0) {
                                    break
                                } else {
                                    return;
                                }
                            }
                        }
                        const inputTypeWidget = this.widgets.find((w) => w.name === 'input_type');
                        inputTypeWidget.value = 'FILE';

                    //  pathWidget.options.values.push(path); // for multiple files
                        pathWidget.value = path;
                        pathWidget.callback?.(path);
                    },
                });
            } else if (type == "file") {
                Object.assign(fileInput, {
                    type: "file",
                    accept: "*.*",
                    style: "display: none",
                    onchange: async () => {
                        if (fileInput.files.length) {
                            const response = await uploadFile(fileInput.files[0])
                            if (response.status != 200) {
                                //upload failed and file can not be added to options
                                return;
                            }
                            const data = await response.json();
                            const inputTypeWidget = this.widgets.find((w) => w.name === 'input_type');
                            inputTypeWidget.value = 'FILE';

                        //  const path = fileInput.files[0].webkitRelativePath;
                            const filename = (data.subfolder ? `${data.subfolder}/` : '') + data.name;
                        //  pathWidget.options.values.push(filename); // for multiple files
                            pathWidget.value = filename;
                            pathWidget.callback?.(filename);
                        }
                    },
                });
            } else {
                throw "Unknown upload type"
            }
            document.body.append(fileInput);
            let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
                //clear the active click event
                app.canvas.node_widget = null
    
                fileInput.click();
            });
            uploadWidget.options.serialize = false;
        };
    },
});