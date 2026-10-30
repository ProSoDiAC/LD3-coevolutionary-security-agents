{{- define "range.podTemplate" }}
      containers:
        - name: app
          image: "{{ .root.Values.image.repository }}:{{ .root.Values.image.tag }}"
          args: ["netexec", "--http-port=8080"]
          ports:
            - containerPort: 8080
          resources:
{{ toYaml .root.Values.resources | indent 12 }}
{{- end -}}
