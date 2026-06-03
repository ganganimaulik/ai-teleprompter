const { execSync } = require('child_process');

module.exports = async function(context) {
  if (process.platform === 'darwin') {
    const artifactPaths = context.artifactPaths || [];
    const dmgPath = artifactPaths.find(p => p.endsWith('.dmg'));
    
    if (dmgPath) {
      console.log(`\n🚀 Opening DMG automatically: ${dmgPath}`);
      execSync(`open "${dmgPath}"`);
    }
  }
};
