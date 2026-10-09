// Material Color Utilities ships ES modules whose relative imports omit ".js". Bundlers accept that;
// plain Node does not, so this resolve hook adds the extension when the token script runs.
export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    if (error.code === "ERR_MODULE_NOT_FOUND" && specifier.startsWith(".") && !specifier.endsWith(".js")) {
      return nextResolve(`${specifier}.js`, context);
    }
    throw error;
  }
}
