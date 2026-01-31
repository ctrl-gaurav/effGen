# Security Policy

## Secure Agent Execution

effGen provides multiple layers of security for running code-executing agents:

### Sandboxed Execution

- **Docker Sandbox**: Isolated container execution with resource limits
- **Memory Limits**: Configurable memory caps
- **Network Isolation**: Optional network disabling
- **Timeout Controls**: Automatic termination of long-running code

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it by:

1. **Email**: gks@vt.edu
2. **GitHub**: Open a private security advisory at [GitHub Security](https://github.com/ctrl-gaurav/effGen/security/advisories)

Please do not publicly disclose the vulnerability until we've had a chance to address it.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.0.x   | Yes       |

## Best Practices

1. Always run untrusted code in Docker sandbox
2. Use API rate limiting
3. Validate all external inputs
4. Keep dependencies updated
