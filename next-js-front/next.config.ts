/** @type {import('next').NextConfig} */
const nextConfig = {
  serverActions: {
    bodySizeLimit: '50mb'
  },
  api: {
    bodyParser: {
      sizeLimit: '50mb'
    }
  }
}

module.exports = nextConfig